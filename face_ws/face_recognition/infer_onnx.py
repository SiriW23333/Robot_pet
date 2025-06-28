import os
import onnxruntime as ort
import numpy as np
import config as FIG
from PIL import Image
import cv2
from scipy.spatial.distance import cosine, chebyshev
import face_sqlite  # 导入数据库操作模块

class create_feature_onnx():
    def __init__(self, model_path, img_path, catch_picture_size, feature_path):
        self.model_path = model_path
        self.img_path = img_path
        self.picture_size = catch_picture_size
        self.feature_path = feature_path

    # 给图片加上黑边框
    def padding_black(self, img):
        w, h = img.size
        scale = self.picture_size / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = self.picture_size
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    # 图像预处理函数 - 替代 torch transforms
    def val_tf(self, image):
        """
        图像预处理，替代 transforms.Compose
        包含：Resize, ToTensor, Normalize
        """
        # 1. Resize - 调整图像大小
        image = image.resize((self.picture_size, self.picture_size))
        
        # 2. ToTensor - 转换为numpy数组并归一化到[0,1]，变换维度
        img_array = np.array(image, dtype=np.float32)  # 确保是float32
        img_array = img_array / 255.0  # 归一化到[0,1]
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        
        # 3. Normalize - 标准化 (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)  # 确保是float32
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)   # 确保是float32
        img_array = (img_array - mean) / std
        
        # 4. 添加batch维度，确保最终结果是float32
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        return img_array

    def save_feature_2(self, featureList, img_name):
        with open(self.feature_path, 'a') as f:
            for i in range(len(featureList)):
                name = img_name[i]
                print(name)
                feature = str(featureList[i])
                f.writelines(name)
                f.writelines(":")
                f.writelines(feature)
                f.writelines('\n')

    # 查找路径下的所有图片
    def search_picutre(self, img_path):
        imgList = []
        img_name = []
        dirs = os.listdir(img_path)
        for file in dirs:
            img_name.append(file)
            pic_dir = os.path.join(img_path, file)
            a = []
            for i in os.listdir(pic_dir):
                img_dir = os.path.join(pic_dir, i)
                a.append(img_dir)
            imgList.append(a)
        return imgList, img_name

    # 主函数
    def feature(self):
        # 加载ONNX模型
        print("Loading ONNX model...")
        session = ort.InferenceSession(self.model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        featurelist = []
        Oneperson_feature = []
        
        # 加载图片
        img_path, img_name = self.search_picutre(self.img_path)
        for pic_path in img_path:
            for image_path in pic_path:
                img = Image.open(image_path)  # 打开图片
                img = img.convert('RGB')  # 转换图片格式
                image = self.padding_black(img)  # 为图片加上黑边
                img_tensor = self.val_tf(image)   # 图片预处理 (不使用torch)

                # 进行ONNX推理
                features = session.run([output_name], {input_name: img_tensor})[0]
                Oneperson_feature.append(features)
                
            featuremean = np.array(Oneperson_feature).mean(axis=0)
            featurelist.append(featuremean.tolist())
            Oneperson_feature.clear()  # 计算完一个人的特征向量后清除
            
        self.save_feature_2(featurelist, img_name)
        print("完成了{}个人的特征向量提取".format(len(featurelist)))

# 全局图像预处理函数
def preprocess_image(image, target_size):
    """
    全局图像预处理函数，不依赖torch transforms
    """
    # 1. Resize
    image = image.resize((target_size, target_size))
    
    # 2. ToTensor
    img_array = np.array(image, dtype=np.float32)  # 确保是float32
    img_array = img_array / 255.0  # 归一化到[0,1]
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    
    # 3. Normalize
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)  # 确保是float32
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)   # 确保是float32
    img_array = (img_array - mean) / std
    
    # 4. 添加batch维度，确保最终结果是float32
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array

def padding_black_global(img, target_size):
    """
    全局padding_black函数
    """
    w, h = img.size
    scale = target_size / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    size_fg = img_fg.size
    size_bg = target_size
    img_bg = Image.new("RGB", (size_bg, size_bg))
    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))
    return img_bg

if __name__ == '__main__':
    # 初始化数据库
    face_sqlite.init_db()
    print("数据库初始化完成")
    
    picture_size = 112
    model_path = './mobile_facenet.onnx'

    # 加载ONNX模型
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 处理两张图片
    img_paths = ['./yqq2.jpg', './wxy.jpg']
    features = []
    face_ids = []  # 存储添加到数据库的face_id

    for i, img_path in enumerate(img_paths):
        print(f"\n处理图片 {i+1}: {img_path}")
        
        img = Image.open(img_path)
        img = img.convert('RGB')
        
        # 使用普通函数进行预处理
        image = padding_black_global(img, picture_size)
        img_tensor = preprocess_image(image, picture_size)
        
        # 进行ONNX推理
        feature = session.run([output_name], {input_name: img_tensor})[0]
        feature_vector = feature.flatten()
        features.append(feature_vector)
        
        print(f"提取到特征向量，维度: {feature_vector.shape}")
        
        
        try:
            user_id = face_sqlite.add_client(feature_vector)
            print(f"新用户添加成功，ID: {user_id}")
            face_ids.append(user_id)
        except Exception as e2:
            print(f"添加用户失败: {e2}")
            face_ids.append(None)

    # 计算两个人脸特征向量的相似度
    if len(features) >= 2:
        cosine_dist = cosine(features[0], features[1])
        chebyshev_dist = chebyshev(features[0], features[1])
        cosine_similarity = 1 - cosine_dist  # 余弦相似度

        print(f"\n===== 人脸比较结果 =====")
        print(f"余弦距离: {cosine_dist:.4f}")
        print(f"余弦相似度: {cosine_similarity:.4f}")
        print(f"切比雪夫距离: {chebyshev_dist:.4f}")
        
        # 判断是否为同一人
        if cosine_similarity > 0.7:
            print("结论: 可能是同一个人")
        else:
            print("结论: 不是同一个人")
    
    # 显示数据库中的记录
    print(f"\n===== 数据库记录 =====")
    for i, face_id in enumerate(face_ids):
        if face_id:
            print(f"图片 {i+1} (ID: {face_id}) 已保存到数据库")
        else:
            print(f"图片 {i+1} 保存失败")
            
    print("\n处理完成！")