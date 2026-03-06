import wall_follow.data_process as data_process
import rclpy

def main(args=None):
    
    rclpy.init(args=args)

    minimal_subscriber = data_process.DataProcess()

    rclpy.spin(minimal_subscriber)
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()