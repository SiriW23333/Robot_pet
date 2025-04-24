from setuptools import setup, find_packages
import os

# 定义数据文件路径（XML/YML/图片等）
data_files = [
    # 包含 haarcascade_frontalface_default.xml
    ('share/face_rec', ['face_rec/haarcascade_frontalface_default.xml']),
    # 包含训练数据文件
    ('share/face_rec/trainer', ['face_rec/trainer/trainer.yml']),
    # 包含数据集目录
    ('share/face_rec/dataset', [os.path.join('face_rec/dataset', f) for f in os.listdir('face_rec/dataset') if f.endswith('.jpg')]),
    # 包含launch文件
    ('share/ament_index/resource_index/packages', ['resource/FaceRec']),
    ('share/face_rec/launch', ['launch/face_launch.py'])
]

# 配置setup参数
setup(
    name='face_rec',
    version='0.0.0',
    packages=find_packages(),
    data_files=data_files,
    install_requires=['setuptools', 'opencv-python>=4.5', 'numpy'],
    entry_points={
        'console_scripts': [
            # 入口点格式：'节点名=包名.模块名:入口函数'
            'face_dataset_node = face_rec.face_dataset_node:main',
            'face_training_node = face_rec.face_training_node:main',
            'face_recognition_node = face_rec.face_recognition_node:main'
        ],
    },
)

