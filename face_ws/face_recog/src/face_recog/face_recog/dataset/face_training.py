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
        self.declare_parameter('prototxt_path', 'deploy.prototxt')
        self.declare_parameter('model_path', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

        self.dataset_path = "/home/sunrise/doraemon/src/face_rec/face_rec/dataset"
        self.trainer_path = self.get_parameter('trainer_path').get_parameter_value().string_value
        self.prototxt_path = "/root/face_ws/F2/model/deploy.prototxt"
        self.model_path = "/root/face_ws/F2/model/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        # Load the DNN model
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)

        # Train the model
        self.train_faces()

    def get_images_and_labels(self, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            try:
                PIL_img = Image.open(image_path).convert('RGB')  # Convert to RGB
                img_numpy = np.array(PIL_img, 'uint8')

                id = int(os.path.split(image_path)[-1].split(".")[1])

                # Prepare the image for the DNN model
                blob = cv2.dnn.blobFromImage(img_numpy, 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.net.setInput(blob)
                detections = self.net.forward()

                # Process detections
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:  # Confidence threshold
                        box = detections[0, 0, i, 3:7] * np.array([img_numpy.shape[1], img_numpy.shape[0],
                                                                    img_numpy.shape[1], img_numpy.shape[0]])
                        (x, y, x1, y1) = box.astype("int")
                        face_samples.append(img_numpy[y:y1, x:x1])
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

        # Convert faces to grayscale and resize for uniformity
        faces = [cv2.cvtColor(face, cv2.COLOR_RGB2GRAY) for face in faces]
        faces = [cv2.resize(face, (200, 200)) for face in faces]

        # Initialize LBPHFaceRecognizer for training
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))

        # Ensure the trainer directory exists
        os.makedirs(os.path.dirname(self.trainer_path), exist_ok=True)

        # Save the model
        recognizer.write(self.trainer_path)
        self.get_logger().info(f"{len(np.unique(ids))} faces trained. Model saved to {self.trainer_path}")


def main(args=None):
    rclpy.init(args=args)
    node = FaceTrainerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
