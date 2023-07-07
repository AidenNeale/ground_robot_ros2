# Import the necessary libraries
import cv2
import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import CompressedImage # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

class ImagePublisher(Node):
  """
  Create an ImagePublisher class, which is a subclass of the Node class. This node's
  purpose is to read and publish camera feed.

  Publisher:
  ----------
  /video_frames: CompressedImage
    ROS2 Compressed Image
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('camera')

    # Create the publisher. This publisher will publish an Image
    # to the video_frames topic. The queue size is 10 messages.
    self.publisher_ = self.create_publisher(CompressedImage, 'frame', 10)

    # We will publish a message every 0.05 seconds
    timer_period = 0.05  # seconds

    # Create the timer
    self.timer = self.create_timer(timer_period, self.timer_callback)

    # Create a VideoCapture object
    # The argument '0' gets the default cam.
    self.cap = cv2.VideoCapture(0)
    # Set X, Y Resolution for camera image
    self.cap.set(3, 640)
    self.cap.set(4, 480)
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()


  def timer_callback(self):
    """
    Callback function.
    This function gets called every 0.01 seconds.
    """
    # Capture frame-by-frame
    # This method returns True/False as well
    # as the video frame.
    ret, frame = self.cap.read()

    if ret == True:
      frame = cv2.rotate(frame, cv2.ROTATE_180) # Flips the image the correct orientation
      # Publish the image.
      # The 'cv2_to_imgmsg' method converts an OpenCV
      # image to a ROS 2 image message
      self.publisher_.publish(self.br.cv2_to_compressed_imgmsg(frame))


def main(args=None):

  # Initialize the rclpy library
  rclpy.init(args=args)

  # Create the node
  image_publisher = ImagePublisher()

  # Spin the node so the callback function is called.
  rclpy.spin(image_publisher)

  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_publisher.destroy_node()

  # Shutdown the ROS client library for Python
  rclpy.shutdown()

if __name__ == '__main__':
  main()