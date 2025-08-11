import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import time
import logging
from typing import List, Dict, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MultiLabelClassifier')

class MultiLabelClassifier:
    def __init__(self, model_path: str, config_path: str, label_list: List[str], device: str = None):
        # 加载配置
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"成功加载配置: {config_path}")
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            raise
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"设备: {self.device}")
        
        # 保存标签列表
        self.label_list = label_list
        self.num_classes = len(label_list)
        logger.info(f"加载 {self.num_classes} 个标签")
        
        try:
            self.model = self._build_model()
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"成功加载模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.config['IMG_SIZE'], self.config['IMG_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config['NORMALIZE_MEAN'], 
                                 std=self.config['NORMALIZE_STD'])
        ])

    def _build_model(self):
        model = models.resnet152(pretrained=False)
        
        # 修改输出层
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, self.num_classes)
        )
        return model

    def predict(self, image_path: str, threshold: float = 0.5, top_k: int = None):
        # 预测单张图像，返回标签和置信度
        start_time = time.time()
        
        try:
            image = Image.open(image_path).convert('RGB')
            logger.debug(f"加载图像: {image_path} ({image.size[0]}x{image.size[1]})")
            original_size = image.size
            
            # 预处理
            image = self.transform(image)
            image = image.unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                output = self.model(image)
                probabilities = torch.sigmoid(output).cpu().numpy().flatten()
            
            if top_k is not None:
                # 获取top_k
                top_indices = np.argsort(probabilities)[::-1][:top_k]
                sorted_labels = [self.label_list[i] for i in top_indices]
                sorted_scores = probabilities[top_indices].tolist()
            else:
                # 获取超过阈值的索引
                above_threshold = np.where(probabilities >= threshold)[0]
                # 按置信度降序排序
                sorted_indices = above_threshold[np.argsort(probabilities[above_threshold])[::-1]]
                sorted_labels = [self.label_list[i] for i in sorted_indices]
                sorted_scores = probabilities[sorted_indices].tolist()
            
            result = {
                'labels': sorted_labels,
                'scores': sorted_scores,
                'all_scores': probabilities.tolist(),
                'original_size': original_size,
                'processed_size': (self.config['IMG_SIZE'], self.config['IMG_SIZE']),
                'time': time.time() - start_time
            }
            
            logger.info(f"预测完成: {image_path} - 耗时 {result['time']:.3f}秒")
            logger.debug(f"Top 3 结果: {sorted_labels[:3]} ({sorted_scores[:3]})")
            return result
        
        except Exception as e:
            logger.error(f"预测图像 {image_path} 失败: {str(e)}")
            return {
                'error': str(e),
                'image_path': image_path
            }

    def predict_batch(self, image_paths: List[str], threshold: float = 0.5, 
                     top_k: int = None, batch_size: int = 8) -> List[Dict]:

        results = []
        num_images = len(image_paths)
        logger.info(f"开始批量预测: {num_images} 张图像, 批量大小 {batch_size}")
        
        # 分批处理
        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_original_sizes = []
            
            # 加载和预处理批量图像
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    batch_original_sizes.append(image.size)
                    image = self.transform(image)
                    batch_images.append(image)
                except Exception as e:
                    logger.error(f"无法加载图像 {path}: {str(e)}")
                    batch_original_sizes.append((0, 0))
                    batch_images.append(torch.zeros(3, self.config['IMG_SIZE'], self.config['IMG_SIZE']))
            
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
            
            # 处理批内每个结果
            for j, probs in enumerate(probabilities):
                if top_k is not None:
                    # 获取top_k索引（已排序）
                    top_indices = np.argsort(probs)[::-1][:top_k]
                    sorted_labels = [self.label_list[idx] for idx in top_indices]
                    sorted_scores = probs[top_indices].tolist()
                else:
                    # 获取超过阈值的索引
                    above_threshold = np.where(probs >= threshold)[0]
                    # 按置信度降序排序
                    sorted_indices = above_threshold[np.argsort(probs[above_threshold])[::-1]]
                    sorted_labels = [self.label_list[idx] for idx in sorted_indices]
                    sorted_scores = probs[sorted_indices].tolist()
                
                result = {
                    'labels': sorted_labels,
                    'scores': sorted_scores,
                    'all_scores': probs.tolist(),
                    'original_size': batch_original_sizes[j],
                    'processed_size': (self.config['IMG_SIZE'], self.config['IMG_SIZE']),
                    'image_path': batch_paths[j]
                }
                
                results.append(result)
        
        logger.info(f"批量预测完成: {num_images} 张图像")
        return results

    def get_label_index(self, label_name: str) -> int:
        try:
            return self.label_list.index(label_name)
        except ValueError:
            logger.warning(f"标签 '{label_name}' 不存在")
            return -1

    def get_label_name(self, index: int) -> str:
        try:
            return self.label_list[index]
        except IndexError:
            logger.warning(f"索引 {index} 超出范围 (0-{len(self.label_list)-1})")
            return None

    def warmup(self, iterations: int = 3) -> None:
        logger.info("开始模型预热")
        dummy_input = torch.randn(1, 3, self.config['IMG_SIZE'], self.config['IMG_SIZE']).to(self.device)
        
        start_time = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        logger.info(f"模型预热完成, 耗时 {time.time() - start_time:.3f}秒")
    
    def get_top_labels(self, all_scores: List[float], n: int = 5) -> List[Tuple[str, float]]:

        # 获取排序后的索引
        sorted_indices = np.argsort(all_scores)[::-1][:n]
        return [(self.label_list[i], all_scores[i]) for i in sorted_indices]

# 这里是配置
MODEL_PATH = "./model/final_model_epoch60.pth"
CONFIG_PATH = "./model/model_config.json"
LABEL_PATH = "./model/label_list.json"

with open(LABEL_PATH, 'r') as f:
    LABEL_LIST = json.load(f)  # 标签列表存储在JSON文件中，格式为列表

#LABEL_LIST = ["label_1", "label_2", "label_3", ...]

# 初始化分类器
classifier = MultiLabelClassifier(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    label_list=LABEL_LIST,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# 单张图像预测 （阈值 or top-k）
img_path = None  # 替换为你的图像路径
result = classifier.predict(img_path, threshold=0.3)
for label, score in zip(result['labels'], result['scores']):
    print(f"- {label}: {score:.4f}")


'''
# 批量预测
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg", ...]
batch_results = classifier.predict_batch(image_paths, top_k=3)

for i, res in enumerate(batch_results):
    print(f"\n图像 {image_paths[i]} 预测结果:")
    for label, score in zip(res['labels'], res['scores']):
        print(f"  {label}: {score:.4f}")

# 获取特定标签信息
label_name = None  # 替换为你想查询的标签名
label_index = classifier.get_label_index(label_name)
if label_index != -1:
    print(f"标签{label_name}在所有标签中的概率: {result['all_scores'][label_index]:.4f}")
'''