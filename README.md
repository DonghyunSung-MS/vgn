# Setup Guide with ROS(Melodic) & Anaconda(python 3.6)

### ROS Dependencies
```bash
(venv) pip install rospkg catkin_tools catkin_pkg
#rospy, rosbsg, *_msgs
(venv) pip install --ignore-installed --extra-index-url https://rospypi.github.io/simple/ rospy rosbag sensor-msgs geometry_msgs visualization_msgs tf2_ros
```