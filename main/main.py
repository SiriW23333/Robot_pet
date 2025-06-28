import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from PerceptionMonitor import PerceptionMonitor
from inference import face_recognization


def main(args=None):
    #client_id = face_recognization()
    client_id = 1  #调试用
    if client_id is None:
        print("人脸识别失败或用户退出。应用将关闭。")
        return
    
    rclpy.init(args=args)
    node = PerceptionMonitor(id=client_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        if node: node.get_logger().error(f"未知错误导致rclpy.spin退出: {e}")
    finally:
        if node :
            node.destroy_node()
            self.get_logger().info("节点已销毁。")
        if rclpy.ok():
            rclpy.shutdown()
        print("应用正在退出。")

if __name__ == '__main__':
    main()
