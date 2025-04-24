from setuptools import setup, find_packages
import os

# 定义数据文件路径（XML/YML/图片等）
data_files = [
    # 包含训练数据文件
    #('share/face_recog/trainer', ['face_recog/trainer/trainer.yml']),
    # 包含数据集目录
    ('share/face_recog/dataset', [os.path.join('face_recog/dataset', f) for f in os.listdir('face_recog/dataset') if f.endswith('.jpg')]),
    # 包含模型文件
    ('share/face_recog/models', [
        'face_recog/res10_300x300_ssd_iter_140000_fp16.caffemodel',
        'face_recog/deploy.prototxt'
    ]),
    # 包含launch文件
   # ('share/ament_index/resource_index/packages', ['resource/face_recog']),
    #('share/face_recog/launch', ['launch/face_launch.py'])
]

# 配置setup参数
setup(
    name='face_recog',
    version='0.0.0',
    packages=find_packages(),
    data_files=data_files,
    install_requires=['setuptools', 'opencv-python>=4.5', 'numpy'],
    entry_points={
        'console_scripts': [
            # 入口点格式：'节点名=包名.模块名:入口函数'
            'face_dataset = face_recog.face_dataset:main',
            'face_training = face_recog.face_training:main',
            'face_recognition_node = face_recog.face_detection_node:main'
        ],
    },
)
