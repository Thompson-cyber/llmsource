# Note: Generate by ChatGPT.
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped

class TFListenerNode(Node):
    def __init__(self):
        super().__init__('tf_listener_node')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically check the transform
        self.timer = self.create_timer(1.0, self.print_transform)

    def print_transform(self):
        try:
            # Get the transform from link1 to link2
            transform = self.tf_buffer.lookup_transform('base_footprint', 'goal_frame', rclpy.time.Time())
            self.get_logger().info('Transform from link1 to link2: \n' + str(transform))
        except Exception as e:
            self.get_logger().warn('Could not find transform from link1 to link2: ' + str(e))


def main(args=None):
    rclpy.init(args=args)
    node = TFListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()