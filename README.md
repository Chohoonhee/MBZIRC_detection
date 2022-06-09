# MBZIRC_detection

This repository contains ROS2 packages for MBZIRC detection algorithms.

**Package Dependencies:**

vision_msgs: https://github.com/Kukanani/vision_msgs/tree/ros2

cv_bridge: https://github.com/ros-perception/vision_opencv/tree/ros2/cv_bridge (May already be present, check by running ros2 pkg list)

Build these packages into your workspace. Make sure ROS2 versions are present.

**Other Dependencies:**

Pytorch and torchvision (if using Jetson, refer: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-6-0-now-available/72048)


**Build and run trt_live_detector:**
Make sure the weights of detection are placed in the ros2_ws/src. T
Then, change the following line (84 line in live_detection/live_detection/live_detection_helper.py) to match the your path.

self.model_path = '/home/vilab/torch_example/src/efficientdet-d0_0_10.pth'

The package can now be built and run. Navigate into your workspace run colcon build --packages-select trt_live_detector

Run both these commands sequentially: source /opt/ros/galactic/setup.bash and . install/setup.bash This will source the terminals.

Next, open the gazebo system, and run the follwoing commands:
ros2 run trt_live_detector trt_detector


The results of the detection are published as Detection2DArray messages. Open a new terminal and source it. Run: ros2 topic echo trt_detection

This will now create a node which carries out faster object detection which is clear from the inference time and is displayed on the terminal as well.

For visualizing the results in RViZ2 set the Fixed Frame as camera_frame. Select the topic trt_detection_image to visualize the detections.
