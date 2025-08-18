import os
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from config import config
from multilabel_classifier import MultiLabelClassifier

app = Flask(__name__)
app.config.from_object(config['default'])

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ImageClassifierApp')

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化模型
classifier = None

def initialize_model():
    global classifier
    if classifier is None:
        logger.info("正在初始化模型...")
        classifier = MultiLabelClassifier(
            model_path=app.config['MODEL_PATH'],
            config_path=app.config['CONFIG_PATH'],
            label_path=app.config['LABEL_PATH'],
            device=app.config['DEVICE']
        )
        classifier.warmup()
        logger.info("模型初始化完成")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 确保模型已初始化
    if classifier is None:
        initialize_model()
    
    # 检查是否上传文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # 直接处理图像数据，无需保存文件
        image = Image.open(file.stream).convert('RGB')
        
        # 获取参数并处理特殊值
        threshold = float(request.form.get('threshold', app.config['API_THRESHOLD']))
        top_k_param = request.form.get('top_k', app.config['API_TOP_K'])
        
        # 处理 top_k 参数的特殊值
        if top_k_param == 'all':
            top_k = 0  # 输出所有满足阈值的标签
        else:
            try:
                top_k = int(top_k_param)
            except ValueError:
                top_k = app.config['API_TOP_K']  # 使用默认值
        
        # 预测
        result = classifier.predict(image, threshold=threshold, top_k=top_k)
        
        if result['success']:
            return jsonify({
                'predictions': [
                    {'label': label, 'score': score} 
                    for label, score in zip(result['labels'], result['scores'])
                ],
                'processing_time': result['processing_time']
            })
        else:
            return jsonify({'error': result['error']}), 500
    
    except Exception as e:
        logger.error(f"预测错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    # 确保模型已初始化
    if classifier is None:
        initialize_model()
    
    # 检查是否上传文件
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'No selected files'}), 400
    
    results = []
    
    try:
        # 获取参数并处理特殊值
        threshold = float(request.form.get('threshold', app.config['API_THRESHOLD']))
        top_k_param = request.form.get('top_k', app.config['API_TOP_K'])
        
        if top_k_param == 'all':
            top_k = 0
        else:
            try:
                top_k = int(top_k_param)
            except ValueError:
                top_k = app.config['API_TOP_K']  # 使用默认值
        
        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'error': 'Invalid file'
                })
                continue
                
            try:
                image = Image.open(file.stream).convert('RGB')
                result = classifier.predict(image, threshold=threshold, top_k=top_k)
                
                if result['success']:
                    results.append({
                        'filename': file.filename,
                        'predictions': [
                            {'label': label, 'score': score} 
                            for label, score in zip(result['labels'], result['scores'])
                        ],
                        'processing_time': result['processing_time']
                    })
                else:
                    results.append({
                        'filename': file.filename,
                        'error': result['error']
                    })
            except Exception as e:
                logger.error(f"处理文件 {file.filename} 失败: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"批量预测错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'device': app.config['DEVICE']
    })

if __name__ == '__main__':
    initialize_model()
    app.run(host='0.0.0.0', port=5000, threaded=True)
