import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os

class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('face_recognition_node')
        self.get_logger().info("Face Recognition Node Started")

        # Load recognizer and cascade
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainer/trainer.yml')
        self.cascadePath = "/home/sunrise/doraemon/src/face_rec/face_rec/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # ID counter and names
        self.names = ['None', 'wxy', 'yqq', 'Ilza', 'Z', 'W']

        # Initialize video capture
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)  # set video width
        self.cam.set(4, 480)  # set video height

        # Define min window size to be recognized as a face
        self.minW = 0.1 * self.cam.get(3)
        self.minH = 0.1 * self.cam.get(4)

        # Timer to periodically process frames
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, img = self.cam.read()
        if not ret:
            self.get_logger().error("Failed to capture frame")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(self.minW), int(self.minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less than 100
            if confidence < 100:
                id = self.names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

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
