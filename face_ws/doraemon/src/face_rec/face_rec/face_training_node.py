'''
Training Multiple Faces stored on a DataBase:
    ==> Each face should have a unique numeric integer ID as 1, 2, 3, etc                       
    ==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
    ==> for using PIL, install pillow library with "pip install pillow"

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18   

Modified for ROS2 compatibility
'''

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from PIL import Image
import os

class FaceTrainerNode(Node):
    def __init__(self):
        super().__init__('face_trainer_node')

        # Declare and get parameters
        self.declare_parameter('dataset_path', 'dataset')
        self.declare_parameter('trainer_path', 'trainer/trainer.yml')
        self.declare_parameter('cascade_path', 'haarcascade_frontalface_default.xml')

        self.dataset_path ="/home/sunrise/doraemon/src/face_rec/face_rec/dataset" 
        self.trainer_path = self.get_parameter('trainer_path').get_parameter_value().string_value
        self.cascade_path ="/home/sunrise/doraemon/src/face_rec/face_rec/haarcascade_frontalface_default.xml"
        # Initialize recognizer and detector
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(self.cascade_path)

        # Train the model
        self.train_faces()

    def get_images_and_labels(self, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            try:
                PIL_img = Image.open(image_path).convert('L')  # Convert to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                id = int(os.path.split(image_path)[-1].split(".")[1])
                faces = self.detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)
            except Exception as e:
                self.get_logger().warn(f"Error processing image {image_path}: {e}")

        return face_samples, ids

    def train_faces(self):
        self.get_logger().info("Training faces. It will take a few seconds. Wait ...")
        faces, ids = self.get_images_and_labels(self.dataset_path)

        if len(faces) == 0:
            self.get_logger().error("No faces found in the dataset. Exiting...")
            return

        self.recognizer.train(faces, np.array(ids))

        # Ensure the trainer directory exists
        os.makedirs(os.path.dirname(self.trainer_path), exist_ok=True)

        # Save the model
        self.recognizer.write(self.trainer_path)
        self.get_logger().info(f"{len(np.unique(ids))} faces trained. Model saved to {self.trainer_path}")


def main(args=None):
    rclpy.init(args=args)
    node = FaceTrainerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
