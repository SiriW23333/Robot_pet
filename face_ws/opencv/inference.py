import onnxruntime as ort
import numpy as np
import cv2
import sys
import serial 
from PIL import Image
import face_sqlite  
import threading
import queue

# 加载ONNX模型
model_path = '/root/Robot_pet/face_ws/opencv/mobile_facenet.onnx'
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
model_input_size = (112, 112)

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 在程序开始时初始化数据库
face_sqlite.init_db()

def detect_face(frame, face_cascade, save_dir="face_images"):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    face_count = len(faces)
    face_images = []
    face_locations = []
    
    if face_count > 0:
        for i, (x, y, w, h) in enumerate(faces):
            face_img = frame[y:y+h, x:x+w]
            face_images.append(face_img)
            face_locations.append((x, y, w, h))
    
    return face_count, face_images, face_locations

def padding_black_cv2(image, target_size):
    """
    使用OpenCV实现padding_black功能
    """
    h, w = image.shape[:2]
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    result = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result

def preprocess_image(image):
    """
    预处理图像以适应ONNX模型输入要求
    """
    padded_image = padding_black_cv2(image, model_input_size[0])
    pil_image = Image.fromarray(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
    pil_image = pil_image.resize(model_input_size)
    
    img_array = np.array(pil_image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    preprocessed_image = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return preprocessed_image

def output(frame):
    """
    使用ONNX模型进行推理
    """
    try:
        features = session.run([output_name], {input_name: frame})[0]
        feature_vector = features.flatten()
        
        if np.allclose(feature_vector, 0):
            print("Warning: Feature vector is zero vector")
            return None
        else:
            return feature_vector
            
    except Exception as e:
        print(f"Error in ONNX inference: {e}")
        return None

def identify_face(feature):
    """
    识别人脸，返回ID和相似度，不询问添加用户
    """
    if feature is None:
        return None, 0.0
        
    try:
        # 查找相似人脸，阈值设为0.6
        similar_faces = face_sqlite.find_similar_face(feature, 0.6)
        
        if similar_faces:
            # 找到相似人脸，选择相似度最高的
            best_match = max(similar_faces, key=lambda x: x['cosine_similarity'])
            face_id = best_match['face_id']
            similarity = best_match['cosine_similarity']
            return face_id, similarity
        else:
            return None, 0.0
            
    except Exception as e:
        print(f"人脸识别过程中出错: {e}")
        return None, 0.0

def face_recognization():
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        print("正在启动摄像头进行人脸识别...")
        print("请面向摄像头，程序将自动识别人脸")
        print("只有连续5帧识别到数据库中的用户才会退出")
        print("按 'q' 键退出，按 'a' 键添加当前最大人脸到数据库")
        
        frame_count = 0
        
        # 用于跟踪已知用户的变量
        known_user_id = None
        known_user_frames = 0
        required_known_frames = 5
        
        # 线程通信
        command_queue = queue.Queue()
        exit_flag = threading.Event()
        current_frame = None
        current_faces = []
        frame_lock = threading.Lock()
        
        def keyboard_listener():
            """键盘监听线程"""
            while not exit_flag.is_set():
                try:
                    key = input().strip().lower()
                    if key == 'q':
                        print("收到退出指令")
                        command_queue.put('quit')
                        exit_flag.set()
                    elif key == 'a':
                        print("收到添加人脸指令")
                        command_queue.put('add_face')
                except EOFError:
                    break
        
        # 启动键盘监听线程
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        print("键盘监听已启动，请在控制台输入命令...")

        while not exit_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame_count += 1
            
            # 调用人脸检测函数
            face_count, face_images, face_locations = detect_face(frame, face_cascade)
            
            # 更新当前帧信息供监听线程使用
            with frame_lock:
                current_frame = frame.copy()
                current_faces = list(zip(face_locations, face_images))
            
            current_known_user = None
            
            if face_count > 0:
                for i, ((x, y, w, h), face_img) in enumerate(zip(face_locations, face_images)):
                    # 预处理人脸图片并计算特征向量
                    preprocessed_image = preprocess_image(face_img)
                    feature = output(preprocessed_image)
                    
                    if feature is not None:
                        # 实时识别人脸
                        user_id, similarity = identify_face(feature)
                        
                        if user_id is not None:
                            # 检测到已知用户
                            current_known_user = user_id
                            
                            # 绘制绿色人脸框
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            
                            # 显示用户ID和相似度
                            label = f"ID: {user_id} ({similarity:.2f})"
                            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            # 更新连续检测计数
                            if known_user_id == user_id:
                                known_user_frames += 1
                            else:
                                known_user_id = user_id
                                known_user_frames = 1
                                print(f"开始跟踪用户 ID: {user_id}")
                                
                            # 显示连续检测帧数
                            frames_text = f"Frames: {known_user_frames}/{required_known_frames}"
                            cv2.putText(frame, frames_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                        else:
                            # 未知用户
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            # 重置已知用户计数
                            known_user_id = None
                            known_user_frames = 0
                    else:
                        # 特征提取失败
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(frame, "Processing...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # 重置已知用户计数
                        known_user_id = None
                        known_user_frames = 0
            else:
                # 没有检测到人脸，重置所有计数
                known_user_frames = 0
                known_user_id = None
                
            # 如果当前帧没有检测到之前的已知用户，重置计数
            if current_known_user != known_user_id and current_known_user is not None:
                known_user_id = current_known_user
                known_user_frames = 1
                print(f"切换到新用户 ID: {current_known_user}")
            elif current_known_user is None and known_user_id is not None:
                # 当前帧没有检测到任何已知用户，重置计数
                known_user_id = None
                known_user_frames = 0
                
            # 显示当前状态
            info_text = f"Faces: {face_count} | Frames: {frame_count}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示当前跟踪状态
            if known_user_id is not None:
                track_text = f"Tracking ID: {known_user_id} | Count: {known_user_frames}/{required_known_frames}"
                cv2.putText(frame, track_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                track_text = "Waiting for known face..."
                cv2.putText(frame, track_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示操作提示
            help_text = "Console: 'q' to quit, 'a' to add largest face"
            cv2.putText(frame, help_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition', frame)
            cv2.waitKey(1)  # 非阻塞等待
            
            # 检查是否连续检测到已知用户5帧
            if known_user_id is not None and known_user_frames >= required_known_frames:
                print(f"\n连续{required_known_frames}帧检测到用户 ID: {known_user_id}，自动退出")
                exit_flag.set()
                cap.release()
                cv2.destroyAllWindows()
                print("摄像头资源已释放")
                return known_user_id
            
            # 处理命令队列
            try:
                while not command_queue.empty():
                    command = command_queue.get_nowait()
                    
                    if command == 'quit':
                        print("执行退出命令")
                        exit_flag.set()
                        cap.release()
                        cv2.destroyAllWindows()
                        return None
                        
                    elif command == 'add_face':
                        # 添加面积最大的人脸
                        with frame_lock:
                            if current_faces:
                                # 找到面积最大的人脸
                                largest_face_idx = 0
                                largest_area = 0
                                
                                for i, ((x, y, w, h), _) in enumerate(current_faces):
                                    area = w * h
                                    if area > largest_area:
                                        largest_area = area
                                        largest_face_idx = i
                                
                                # 获取最大人脸
                                (x, y, w, h), face_img = current_faces[largest_face_idx]
                                
                                # 预处理并提取特征
                                preprocessed_image = preprocess_image(face_img)
                                feature = output(preprocessed_image)
                                
                                if feature is not None:
                                    try:
                                        # 添加到数据库
                                        new_user_id = face_sqlite.add_client(feature)
                                        print(f"成功添加新用户，ID: {new_user_id}")
                                        
                                        # 在当前帧上标记添加的人脸
                                        cv2.rectangle(current_frame, (x, y), (x+w, y+h), (255, 0, 255), 3)
                                        cv2.putText(current_frame, f"Added ID: {new_user_id}", (x, y-10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                                        cv2.imshow('Face Recognition', current_frame)
                                        cv2.waitKey(2000)  # 显示2秒
                                        
                                        exit_flag.set()
                                        cap.release()
                                        cv2.destroyAllWindows()
                                        print("摄像头资源已释放")
                                        return new_user_id
                                        
                                    except Exception as e:
                                        print(f"添加用户失败: {e}")
                                else:
                                    print("无法提取人脸特征，添加失败")
                            else:
                                print("当前没有检测到人脸，无法添加")
                                
            except queue.Empty:
                pass

        # 正常情况下不会到达这里
        cap.release()
        cv2.destroyAllWindows()
        return None
    except Exception as e:
        print(f"人脸识别过程中出错: {e}")
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头资源已释放")
        return None
  


