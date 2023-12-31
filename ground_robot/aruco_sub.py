# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

from cv2 import aruco
 
class ArucoNode(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('aruco_node')
      
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(
      Image, 
      'video_frames', 
      self.listener_callback, 
      10)
    self.subscription # prevent unused variable warning
      
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()

    # self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    # self.aruco_param = aruco.DetectorParameters_create()

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters =  aruco.DetectorParameters()
    self.detector_aruco = aruco.ArucoDetector(dictionary, parameters)

    # Publisher

    # Add custom msg -> self.publisher_aruco_coords = self.create_publisher(????, 'frame', 10)


  def listener_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    # self.get_logger().info('Receiving video frame')
 
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.imgmsg_to_cv2(data)

    self.find_aruco_tags(current_frame)
    
    # Display image
    cv2.imshow("camera", current_frame)
    
    cv2.waitKey(1)



  def find_aruco_tags(self, img):
    """
    Find ARUCO tags from camera frame
    """

    markerCorners, markerIds, rejectedCandidates = self.detector_aruco.detectMarkers(img)

    # corners,ids,rejectedImgPoints = aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_parameters)
    frame_markers = aruco.drawDetectedMarkers(img, markerCorners, markerIds)

  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  aruco_node = ArucoNode()
  
  # Spin the node so the callback function is called.
  rclpy.spin(aruco_node)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  aruco_node.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
