import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

class FaceCaptureNode(Node):
    def __init__(self):
        super().__init__('face_capture_node')
        self.bridge = CvBridge()
        self.cam = cv2.VideoCapture(0)

        self.cam.set(3, 640)  # 设置摄像头宽度
        self.cam.set(4, 480)  # 设置摄像头高度
        
        # 初始化DNN模型
        self.net = self.load_dnn_model()
        self.conf_threshold = 0.7  # 置信度阈值
        
        self.image_pub = self.create_publisher(Image, 'captured_image', 10)
        self.face_id = None
        self.count = 0

    def load_dnn_model(self):
        # 模型路径配置
        framework = "caffe"  # 可选"caffe"或"tf"
        
        if framework == "caffe":
            modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "deploy.prototxt"
            net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        else:
            modelFile = "opencv_face_detector_uint8.pb"
            configFile = "opencv_face_detector.pbtxt"
            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        
        # 设置计算设备（根据开发板支持情况选择）
        # 如果开发板支持BPU（如某些AI芯片），需要使用对应的SDK或API加载模型
        # 这里假设使用OpenCV DNN的默认设置（CPU）
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def detect_faces(self, frame):
        frameHeight = frame.shape
        frameWidth = frame.shape
        
        # 创建输入blob
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False
        )
        
        # 前向推理
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # 解析检测结果
        bboxes = []
        for i in range(detections.shape):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append((x1, y1, x2-x1, y2-y1))  # 转换为(x,y,w,h)格式
        return bboxes

    def start_capture(self, face_id):
        self.face_id = face_id
        self.count = 0
        self.get_logger().info("Starting face capture. Look at the camera...")
        
        while rclpy.ok() and self.count < 30:
            ret, frame = self.cam.read()
            if not ret:
                self.get_logger().error("Camera read error")
                break

            # 人脸检测
            bboxes = self.detect_faces(frame)
            
            # 处理检测结果
            for (x, y, w, h) in bboxes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 保存人脸区域
                self.count += 1
                dataset_path = "/root/face_ws/F2/model/dataset"
                os.makedirs(dataset_path, exist_ok=True)
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(f"{dataset_path}/User.{self.face_id}.{self.count}.jpg", face_img)
                
                # 发布图像消息
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.image_pub.publish(ros_image)
            
            # 显示实时画面
            cv2.imshow('Face Capture', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                break

        # 清理资源
        self.cam.release()
        cv2.destroyAllWindows()
        self.get_logger().info(f"Captured {self.count} samples for user {self.face_id}")

def main(args=None):
    rclpy.init(args=args)
    node = FaceCaptureNode()
    face_id = input("Enter user ID: ").strip()
    node.start_capture(face_id)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
