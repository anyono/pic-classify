import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from skmultilearn.model_selection import iterative_train_test_split # scikit-multilearn
import pandas as pd
from PIL import Image
import os
import numpy as np
# from sklearn.model_selection import train_test_split
from collections import OrderedDict
import gc
import signal
import re
import sys
from sklearn.metrics import f1_score, average_precision_score

class Config:
    DATA_DIR = "DIR_TO_IMAGE"
    CSV_PATH = "DIR_OF_LABEL"
    IMG_SIZE = 512
    BATCH_SIZE = 8
    NUM_EPOCHS = 60
    NUM_CLASSES = -1 # 手动输入数据集的标签数
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = "DIR_TO_SAVE_MODEL_AND_CHECKPOINT"

    # 解冻参数
    FREEZE_EPOCHS = 10
    UNFREEZE_50_PERCENT = 20
    UNFREEZE_ALL = 40

    # 梯度累计
    GRAD_ACCUM_STEPS = 4

config = Config()

# 多标签数据集类
class MultiLabelDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.image_names = df.iloc[:, 0].values

        # 标签从第二列开始
        self.labels = df.iloc[:, 1:].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, torch.tensor(label)

# 数据变换
def get_transforms(phase='train'):

    # 来自ImageNet的标准化参数，基于ResNet-152迁移学习
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

# 加载数据集
def prepare_datasets():
    df = pd.read_csv(config.CSV_PATH)
    X = df.iloc[:, 0:1].values
    y = df.iloc[:, 1:].values

    # 训练集占80%
    X_train, y_train, X_val, y_val = iterative_train_test_split(X, y, test_size=0.2)

    train_df = pd.DataFrame(np.concatenate((X_train, y_train), axis=1), columns=df.columns)
    val_df = pd.DataFrame(np.concatenate((X_val, y_val), axis=1), columns=df.columns)

    train_dataset = MultiLabelDataset(train_df, config.DATA_DIR, transform=get_transforms('train'))
    val_dataset = MultiLabelDataset(val_df, config.DATA_DIR, transform=get_transforms('val'))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# 创建模型
def create_model():
    model = models.resnet152(pretrained=True)
    layers = OrderedDict([
        ('conv1', model.conv1),
        ('bn1', model.bn1),
        ('relu', model.relu),
        ('maxpool', model.maxpool),
        ('layer1', model.layer1),
        ('layer2', model.layer2),
        ('layer3', model.layer3),
        ('layer4', model.layer4),
        ('avgpool', model.avgpool)
    ])

    # total_layers = len(layers)
    # unfreeze_start = total_layers - int(np.ceil(total_layers * 0.5))

    # 冻结所有层
    for i, (name, layer) in enumerate(layers.items()):
        for param in layer.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, config.NUM_CLASSES)
    )

    # 解冻fc层
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.AdamW(trainable_params, lr=config.LR)
    
    model = model.to(config.DEVICE)
    return model, layers, criterion, optimizer

# 训练函数
def train_model(model, layers, criterion, optimizer, train_loader, val_loader, start_epoch=0, best_val_loss=float('inf')):
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(config.SAVE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    freeze_state = 0
    if start_epoch >= config.UNFREEZE_ALL:
        freeze_state = 2
    elif start_epoch >= config.UNFREEZE_50_PERCENT:
        freeze_state = 1

    # 手动中断检查点
    def handle_interrupt(signal, frame):
        nonlocal epoch, best_val_loss, freeze_state
        print("\n\n捕捉到中断信号,保存当前状态")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'freeze_state': freeze_state,
            'config': vars(config)
        }

        interrupt_path = os.path.join(checkpoint_dir, f'interrupt_checkpoint_epoch{epoch+1}.pth')
        torch.save(checkpoint, interrupt_path)
        print(f"已保存中断检查点: {interrupt_path}")

        gc.collect()
        torch.cuda.empty_cache()
        sys.exit(0)

    # 注册中断信号处理
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    try:
        epoch = start_epoch - 1
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS} 开始")

            trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
            print(f"Epoch {epoch + 1} - Trainable parameters: {len(trainable_params)}")
            print("可训练参数总数:", sum(p.numel() for p in model.parameters() if p.requires_grad))

            def rebuild_optimizer_scheduler(model, optimizer, scheduler):
                current_lr = optimizer.param_groups[0]['lr']
                trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
                optimizer = optim.AdamW(trainable_params, lr=current_lr)
                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
                return optimizer, scheduler

            # 解冻layer3, layer4, avgpool
            if freeze_state < 1 and epoch >= config.UNFREEZE_50_PERCENT:
                print(f"Epoch {epoch + 1}: 解冻50%层")
                total_layers = len(layers)
                unfreeze_start = total_layers - int(np.ceil(total_layers * 0.5))

                for i, (name, layer) in enumerate(layers.items()):
                    for param in layer.parameters():
                        param.requires_grad = (i >= unfreeze_start)
                
                freeze_state = 1
                optimizer, scheduler = rebuild_optimizer_scheduler(model, optimizer, scheduler)
            
            # 解冻所有层
            elif freeze_state < 2 and epoch >= config.UNFREEZE_ALL:
                print(f"Epoch {epoch+1}: 解冻所有层")
                for name, layer in layers.items():
                    for param in layer.parameters():
                        param.requires_grad = True
                
                freeze_state = 2
                optimizer, scheduler = rebuild_optimizer_scheduler(model, optimizer, scheduler)

            print("冻结状态:")
            for name, layer in layers.items():
                print(f"{name}: {'解冻' if any(p.requires_grad for p in layer.parameters()) else '冻结'}")

            # 训练阶段
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            for step, (inputs, labels) in enumerate(train_loader):
                if step % 50 == 0:
                    print(f"Epoch {epoch+1} - Step {step+1}/{len(train_loader)}")

                if step % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                inputs = inputs.to(config.DEVICE, non_blocking=True)
                labels = labels.to(config.DEVICE, non_blocking=True)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 梯度累积
                loss = loss / config.GRAD_ACCUM_STEPS
                loss.backward()

                # 累积损失
                running_loss += loss.item() * inputs.size(0) * config.GRAD_ACCUM_STEPS

                # 每隔GRAD_ACCUM_STEPS步更新一次参数
                if (step + 1) % config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            # 计算每个epoch的平均损失
            epoch_loss = running_loss / len(train_loader.dataset)

            # 验证阶段
            val_loss = evaluate(model, criterion, val_loader)
            print(f'Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}')

            # 更新学习率
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(config.SAVE_DIR, f'best_model_epoch{epoch+1}_valloss{val_loss:.4f}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"保存最佳模型: {model_path}")

            # 检查点
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'freeze_state': freeze_state,
                'config': vars(config)
            }

            # 保存每epoch检查点
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")

            # 保存最新检查点
            latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            torch.save(checkpoint, latest_path)

            # 清理内存
            gc.collect()
            torch.cuda.empty_cache()

        print("训练完成！")

        final_model_path = os.path.join(config.SAVE_DIR, f'final_model_epoch{config.NUM_EPOCHS}.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"保存最终模型: {final_model_path}")

        return best_val_loss
    
    except Exception as e:
        print(f"error: {e}")
        print("保存当前状态")

        # 保存error检查点
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'freeze_state': freeze_state,
                'config': vars(config)
            }

            error_path = os.path.join(checkpoint_dir, f'error_checkpoint_epoch{epoch+1}.pth')
            torch.save(checkpoint, error_path)
            print(f"已保存检查点: {error_path}")
        except Exception as save_error:
            print(f"保存检查点时出错: {save_error}")

        # 抛出异常
        raise e

# 验证函数
def evaluate(model, criterion, data_loader):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # 预测概率
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    torch.cuda.empty_cache()

    # 连接所有batch的预测和标签
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # 二值化预测
    binarized_preds = (all_preds > 0.5).astype(int)

    # 计算F1分数和mAP
    f1 = f1_score(all_targets, binarized_preds, average='micro', zero_division=0)
    map_score = average_precision_score(all_targets, all_preds, average='micro')

    avg_loss = running_loss / len(data_loader.dataset)
    print(f"Val Loss: {avg_loss:.4f} | F1 (micro): {f1:.4f} | mAP: {map_score:.4f}")
    return avg_loss


if __name__ == "__main__":
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    train_loader, val_loader = prepare_datasets()

    # 检查点目录
    checkpoint_dir = os.path.join(Config.SAVE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_ckpt = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

    # 恢复顺序：latest > checkpoint_epoch > interrupt/error
    if os.path.exists(latest_ckpt):
        print(f"发现最新检查点: {latest_ckpt}, 恢复训练")
        checkpoint = torch.load(latest_ckpt)

    elif any(f.startswith('checkpoint_epoch') for f in os.listdir(checkpoint_dir)):
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch')]

        def extract_epoch(filename):
            match = re.search(r'epoch(\d+)', filename)
            return int(match.group(1)) if match else 0

        ckpt_files.sort(key=extract_epoch)
        latest_ckpt_path = os.path.join(checkpoint_dir, ckpt_files[-1])
        print(f"发现历史检查点: {latest_ckpt_path}, 恢复训练")
        checkpoint = torch.load(latest_ckpt_path)

    elif any('interrupt' in f or 'error' in f for f in os.listdir(checkpoint_dir)):
        inter_files = [f for f in os.listdir(checkpoint_dir) if 'interrupt' in f or 'error' in f]

        def extract_epoch(filename):
            import re
            match = re.search(r'epoch(\d+)', filename)
            return int(match.group(1)) if match else 0

        inter_files.sort(key=extract_epoch)
        latest_interrupt = os.path.join(checkpoint_dir, inter_files[-1])
        print(f"发现中断或错误检查点: {latest_interrupt}, 恢复训练")
        checkpoint = torch.load(latest_interrupt)

    else:
        checkpoint = None
        print("未发现任何检查点，从头开始训练")

    # 如果 checkpoint 被加载成功
    if checkpoint:
        model, layers, criterion, optimizer = create_model()
        model.load_state_dict(checkpoint['model_state_dict'])

        freeze_state = checkpoint.get('freeze_state', 0)

        def restore_freeze_state(layers, freeze_state):
            total_layers = len(layers)
            unfreeze_start = total_layers - int(np.ceil(total_layers * 0.5))
            for i, (name, layer) in enumerate(layers.items()):
                # 根据冻结状态设置参数的 requires_grad
                if freeze_state == 0:
                    for param in layer.parameters():
                        param.requires_grad = (i >= unfreeze_start)

                elif freeze_state == 1:
                    for param in layer.parameters():
                        param.requires_grad = (i >= unfreeze_start)
                elif freeze_state == 2:
                    for param in layer.parameters():
                        param.requires_grad = True

        # 恢复冻结状态
        restore_freeze_state(layers, freeze_state)

        print("恢复后冻结状态:")
        for name, layer in layers.items():
            print(f"{name}: {'解冻' if any(p.requires_grad for p in layer.parameters()) else '冻结'}")

        checkpoint_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = optim.AdamW(trainable_params, lr=checkpoint_lr)

        # 恢复optimizer状态
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print(f"error:优化器状态恢复失败，使用新初始化的优化器状态: {e}")

        # 打印optimizer状态
        print(f"优化器参数组数量: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            print(f"参数组 {i+1}: 学习率={group['lr']}, 参数数量={len(group['params'])}")
        print("\n\n")

        # 恢复scheduler状态
        scheduler_state = checkpoint.get('scheduler_state_dict', None)
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if scheduler_state is not None:
            print("恢复scheduler状态")
            # 创建新的调度器
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
            scheduler.load_state_dict(scheduler_state)
        else:
            print("未找到scheduler状态,使用默认配置")
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

        # 打印模型参数状态
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if param.requires_grad:
                trainable_params += 1
                print(f"[可训练] {name}")
            else:
                print(f"[冻结] {name}")

        print(f"参数状态: {trainable_params}/{total_params} 可训练")
        print(f"可训练参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print("="*40)

        print(f"恢复训练: epoch {start_epoch}, 最佳val loss: {best_val_loss:.4f}")
        train_model(model, layers, criterion, optimizer, train_loader, val_loader,
                    start_epoch=start_epoch, best_val_loss=best_val_loss)

    else:
        print("未发现检查点，开始新训练")
        model, layers, criterion, optimizer = create_model()

        print("初始冻结状态:")
        for name, layer in layers.items():
            print(f"{name}: {'解冻' if any(p.requires_grad for p in layer.parameters()) else '冻结'}")

        train_model(model, layers, criterion, optimizer, train_loader, val_loader)
    config_data = {
        'IMG_SIZE': config.IMG_SIZE,
        'NORMALIZE_MEAN': [0.485, 0.456, 0.406],
        'NORMALIZE_STD': [0.229, 0.224, 0.225],
        'MODEL_TYPE': 'resnet152',
        'NUM_CLASSES': config.NUM_CLASSES
    }
    
    config_path = os.path.join(config.SAVE_DIR, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    # 标签列表（csv）
    df = pd.read_csv(config.CSV_PATH)
    label_list = df.columns[1:].tolist()  # 第一列是文件名
    
    label_path = os.path.join(config.SAVE_DIR, "label_list.json")
    with open(label_path, 'w') as f:
        json.dump(label_list, f)
    
    print(f"模型配置已保存至: {config_path}")

    print(f"标签列表已保存至: {label_path}")

