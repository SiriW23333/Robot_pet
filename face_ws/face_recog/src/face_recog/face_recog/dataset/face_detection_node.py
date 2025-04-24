import rclpy
from rclpy.node import Node
import cv2
import numpy as np

class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('face_recognition_node')
        self.get_logger().info("Face Recognition Node Started")

        # Load DNN model
        self.model_path = {
            "prototxt": "deploy.prototxt",
            "caffemodel": "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        }
        self.net = cv2.dnn.readNetFromCaffe(self.model_path["prototxt"], self.model_path["caffemodel"])

        # Initialize video capture
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)  # set video width
        self.cam.set(4, 480)  # set video height

        # Timer to periodically process frames
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            self.get_logger().error("Failed to capture frame")
            return

        # Prepare the frame for DNN model
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Perform face detection
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box and confidence
                text = f"{confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow('camera', frame)

        k = cv2.waitKey(10) & 0xff
        if k == 27:  # Press 'ESC' to exit
            self.get_logger().info("Exiting Program and cleanup stuff")
            self.destroy_node()
            rclpy.shutdown()
            self.cam.release()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
