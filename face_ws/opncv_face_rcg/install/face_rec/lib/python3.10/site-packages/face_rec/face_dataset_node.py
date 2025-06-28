''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    

'''


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class FaceCaptureNode(Node):
    def __init__(self):
        super().__init__('face_capture_node')
        self.bridge = CvBridge()
        self.cam = cv2.VideoCapture(1)
        self.cam.set(3, 640)  # set video width
        self.cam.set(4, 480)  # set video height
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.image_pub = self.create_publisher(Image, 'captured_image', 10)
        self.face_id = None
        self.count = 0

    def start_capture(self, face_id):
        self.face_id = face_id
        self.count = 0
        self.get_logger().info("Initializing face capture. Look at the camera and wait...")
        while rclpy.ok():
            ret, img = self.cam.read()
            if not ret:
                self.get_logger().error("Failed to capture image from camera.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                self.count += 1

                # Save the captured image into the dataset folder
                dataset_path = os.path.join(os.getcwd(), 'dataset')
                os.makedirs(dataset_path, exist_ok=True)
                file_path = os.path.join(dataset_path, f"User.{self.face_id}.{self.count}.jpg")
                cv2.imwrite(file_path, gray[y:y + h, x:x + w])

                # Publish the image
                ros_image = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                self.image_pub.publish(ros_image)

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27 or self.count >= 30:  # Stop after 30 samples or ESC key
                break

        self.get_logger().info("Exiting face capture and cleaning up.")
        self.cam.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = FaceCaptureNode()
    face_id = input('\nEnter user ID and press <return>: ')
    node.start_capture(face_id)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

