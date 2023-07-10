# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

import imutils
import numpy as np
import pytesseract
 
class PlateNode(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('plate_subs')
      
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
   
  def listener_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    # self.get_logger().info('Receiving video frame')
 
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.imgmsg_to_cv2(data)

    self.find_number_plate(current_frame)
    
    # Display image
    cv2.imshow("camera", current_frame)
    
    cv2.waitKey(1)



  def find_number_plate(self, img):
    """
    Find number plate from camera frame
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15) 

    # Edge and contour detection
    edged = cv2.Canny(gray, 30, 200) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    cntList = []

    # Check contours for closed rectangles
    for c in contours:
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        
        # Check if rectangle 
        if len(approx) == 4:
            cntList.append(approx)

    # Check for rectange detection
    if len(cntList) != 0:
        # print(f"Number of possible plates detected: {len(cntList)}")
        # self.get_logger().info(f"Number of possible plates detected: {len(cntList)}")

        # Draw all possiblities
        # for i in range(len(cntList)):
        #     cv2.drawContours(img, [cntList[i]], -1, (0, 0, 255), 3)

        # Check each possible plate
        for screenCnt in cntList:

            # Mask off just plate
            mask = np.zeros(gray.shape,np.uint8)
            new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
            new_image = cv2.bitwise_and(img,img,mask=mask)

            # Crop down to just plate
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx+1, topy:bottomy+1]

            # OCR on plate
            text = pytesseract.image_to_string(Cropped, config='--psm 10')

            # Check for 'york' in plate text. Hasn't in testing, but could lead to problems between 0 and O.
            if 'york' in text.lower():

                # print("License plate detected, number is:",text)
                cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
                self.get_logger().info(f"License plate detected, number is:{text}")
                
                # Display results
                # img_display = cv2.resize(img,(500,300))
                # Cropped = cv2.resize(Cropped,(400,200))
                # cv2.imshow('Car',img_display)
                # cv2.imshow('Cropped',Cropped)

                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                break
            else:
                # print("No licence plate detected")
                self.get_logger().info("No license plate detected")
    else:
        self.get_logger().info("No license plate detected")


  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  plate_subs = PlateNode()
  
  # Spin the node so the callback function is called.
  rclpy.spin(plate_subs)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  plate_subs.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
