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
    def __init__(self, model_path: str, config_path: str, label_path: str, device: str = None):
        # 加载标签列表
        try:
            with open(label_path, 'r') as f:
                self.label_list = json.load(f)
            self.num_classes = len(self.label_list)
            logger.info(f"成功加载标签: {label_path}, 共 {self.num_classes} 个标签")
        except Exception as e:
            logger.error(f"加载标签失败: {str(e)}")
            raise
        
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
        
        # 加载模型
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
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.config['IMG_SIZE'], self.config['IMG_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config['NORMALIZE_MEAN'], 
                                 std=self.config['NORMALIZE_STD'])
        ])

    # 构建模型
    def _build_model(self):
        model = models.resnet152(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, self.num_classes)
        )
        return model
    
    # 预测单张图像
    def predict(self, image: Image.Image, threshold: float = 0.5, top_k: int = None):
        start_time = time.time()
        original_size = image.size
        
        try:
            # 预处理并预测
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.sigmoid(output).cpu().numpy().flatten()
            
            # 根据阈值或top_k筛选标签
            if top_k is not None and top_k > 0:
                top_indices = np.argsort(probabilities)[::-1][:top_k]
                sorted_labels = [self.label_list[i] for i in top_indices]
                sorted_scores = probabilities[top_indices].tolist()
            else: # 使用阈值筛选
                above_threshold = np.where(probabilities >= threshold)[0]
                # 按置信度降序排序
                sorted_indices = above_threshold[np.argsort(probabilities[above_threshold])[::-1]]
                sorted_labels = [self.label_list[i] for i in sorted_indices]
                sorted_scores = probabilities[sorted_indices].tolist()
            
            return {
                'success': True,
                'labels': sorted_labels,
                'scores': sorted_scores,
                'all_scores': probabilities.tolist(),
                'original_size': original_size,
                'processing_time': time.time() - start_time
            }
        
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def warmup(self):
        logger.info("开始模型预热")
        dummy_input = torch.randn(1, 3, self.config['IMG_SIZE'], self.config['IMG_SIZE']).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        logger.info("模型预热完成")


if __name__ == "__main__":
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
