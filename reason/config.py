import os
import torch

class Config:
    DEBUG = False
    TESTING = False
    SECRET_KEY = None
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # 模型路径
    MODEL_DIR = 'model'
    MODEL_PATH = os.path.join(MODEL_DIR, 'final_model_epoch60.pth')
    CONFIG_PATH = os.path.join(MODEL_DIR, 'model_config.json')
    LABEL_PATH = os.path.join(MODEL_DIR, 'label_list.json')
    
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # API配置
    API_THRESHOLD = 0.5
    API_TOP_K = 10

# 开发环境配置
class DevelopmentConfig(Config):
    DEBUG = True

# 生产环境配置
class ProductionConfig(Config):
    DEBUG = False

# 配置映射
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
