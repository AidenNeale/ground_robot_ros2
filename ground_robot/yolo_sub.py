# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

from cv2 import aruco
import numpy as np
 
class ImageSubscriber(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_subscriber')
      
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

    # ----------------------------------------------------------

    classes_dir = r"/home/will/ros2_ws/src/cv_basics/cv_basics/yolov3.txt"
    weights_dir = r"/home/will/ros2_ws/src/cv_basics/cv_basics/yolov3.weights"
    config_dir = r"/home/will/ros2_ws/src/cv_basics/cv_basics/yolov3.cfg"

    classes = None

    with open(classes_dir, 'r') as f:
      self.classes = [line.strip() for line in f.readlines()]
    self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
    self.net = cv2.dnn.readNet(weights_dir, config_dir)
    
  def listener_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    # self.get_logger().info('Receiving video frame')
 
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.imgmsg_to_cv2(data)

    self.find_bear(current_frame)
    
    # Display image
    cv2.imshow("camera", current_frame)
    
    cv2.waitKey(1)

  def get_output_layers(self, net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


  def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(self.classes[class_id])
    color = self.COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label+" "+str(confidence), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



  def find_bear(self, img):
    """
    Find the bear from camera frame
    """

    image = img

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    self.net.setInput(blob)

    outs = self.net.forward(self.get_output_layers(self.net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        self.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    # image = cv2.resize(image,(500,300))
    # cv2.imshow("object detection", image)
    # cv2.waitKey()
    
    # cv2.imwrite("object-detection.jpg", image)
    # cv2.destroyAllWindows()

  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  image_subscriber = ImageSubscriber()
  
  # Spin the node so the callback function is called.
  rclpy.spin(image_subscriber)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_subscriber.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
