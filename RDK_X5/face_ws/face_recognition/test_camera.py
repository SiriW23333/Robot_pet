from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy
import numpy as np
import cv2
import sys
import serial 
import os

#create model object
try:
    models = dnn.load('./mobilefacenet.bin')
    model_input_size = (112, 112)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_save_face(frame, face_cascade, save_dir="face_images"):
    """
    检测人脸并保存人脸框内部的图像
    
    Args:
        frame: 输入的图像帧
        face_cascade: 人脸分类器
        save_dir: 保存图片的目录
    
    Returns:
        face_count: 检测到的人脸数量
        face_images: 截取的人脸图片列表
        face_locations: 人脸位置信息列表
        saved_paths: 保存的文件路径列表
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    face_count = len(faces)
    face_images = []
    face_locations = []
    saved_paths = []
    
    if face_count > 0:
        for i, (x, y, w, h) in enumerate(faces):
            # 截取人脸区域
            face_img = frame[y:y+h, x:x+w]
            face_images.append(face_img)
            face_locations.append((x, y, w, h))
            
            # 保存人脸图片
            filename = f"face_{i+1}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, face_img)
            saved_paths.append(filepath)
            print(f"保存人脸图片: {filepath}")
    
    return face_count, face_images, face_locations, saved_paths

def preprocess_image(image):
    """
    预处理图像以适应模型输入要求
    按照 PyTorch MobileFaceNet 的预处理流程
    """
    if image is None:
        print("Error: Input image is None")
        return None
        
    print(f"Input image shape: {image.shape}")
    
    # 调整图像大小
    resized_image = cv2.resize(image, model_input_size)
    
    # 转换颜色空间从BGR到RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # 转换为float32并归一化到[0,1] (相当于 ToTensor())
    float_image = rgb_image.astype(np.float32) / 255.0
    
    # 标准化 (相当于 Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    # 这会将 [0,1] 的值转换为 [-1,1] 的值
    normalized_image = (float_image - 0.5) / 0.5
    
    # 转换为NCHW格式 (相当于 permute(2, 0, 1))
    nchw_image = np.transpose(normalized_image, (2, 0, 1))
    
    # 添加批次维度 (相当于 unsqueeze(0))
    preprocessed_image = np.expand_dims(nchw_image, axis=0)
    
    print(f"Preprocessed image shape: {preprocessed_image.shape}")
    print(f"Preprocessed image dtype: {preprocessed_image.dtype}")
    
    return preprocessed_image

def output(frame):
    """
    处理图像并返回特征向量，同时打印output.buffer
    """
    try:
        outputs = models[0].forward(frame)
        
        # 打印 outputs.buffer
        for i, output_tensor in enumerate(outputs):
            print(f"Output {i} buffer shape: {output_tensor.buffer.shape}")
        
        # 检查是否为零向量
        feature_vector = outputs[0].buffer.copy()  # 创建副本避免内存问题
        if np.allclose(feature_vector, 0):
            print("Warning: Feature vector is zero vector")
            return None
        else:
            print("Feature vector is valid (non-zero)")
            return feature_vector
    except Exception as e:
        print(f"Error in model forward pass: {e}")
        return None

def main():
    '''cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # 调用人脸检测和保存函数
        face_count, face_images, face_locations, saved_paths = detect_and_save_face(frame, face_cascade)
        
        if face_count > 0:
            print(f"检测到 {face_count} 张人脸")
            
            for i, ((x, y, w, h), face_img, saved_path) in enumerate(zip(face_locations, face_images, saved_paths)):
                # 绘制人脸框
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                print(f"\n处理第 {i+1} 张人脸图像: {saved_path}")
                
                # 预处理人脸图片并投入output函数
                preprocessed_image = preprocess_image(face_img)
                feature = output(preprocessed_image)
                
                if feature is not None:
                    print(f"特征向量形状: {feature.shape}")
                    print(f"特征向量前10个值: {feature[:10]}")
        
        # 显示帧数和人脸数量信息
        info_text = f"Faces: {face_count}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
        # 显示视频流
        cv2.imshow('Face Recognition', frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()'''
    # Load and preprocess the images
    img1 = cv2.imread('./cyr.jpg')
    img2 = cv2.imread('./yqq_1.jpg')
    
    if img1 is None or img2 is None:
        print("Error: Could not load one or both images")
        return
    
    preprocessed_img1 = preprocess_image(img1)
    preprocessed_img2 = preprocess_image(img2)
    
    f1 = output(preprocessed_img1)
    f2 = output(preprocessed_img2)
    if f1 is not None and f2 is not None:
        # 计算两个特征向量的余弦相似度
        dot_product = np.dot(f1.flatten(), f2.flatten())
        norm_f1 = np.linalg.norm(f1.flatten())
        norm_f2 = np.linalg.norm(f2.flatten())
        cosine_similarity = dot_product / (norm_f1 * norm_f2)
        
        # 计算欧氏距离
        euclidean_distance = np.linalg.norm(f1.flatten() - f2.flatten())
        
        print(f"\n人脸特征向量比较结果:")
        print(f"余弦相似度: {cosine_similarity}")
        print(f"欧氏距离: {euclidean_distance}")
 

if __name__ == "__main__":
    main()
