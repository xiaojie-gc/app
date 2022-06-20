# Components

1. URB: urb-OFDMA.py
2. Edge server controller: server-OFDMA.py
3. Client: 2022_exp_main.py

Don't forget to change the IP address of the URB.  You need to change the settings both in "server-OFDMA.py" and "2022_exp_main.py".

You can run #1 and #2 on the same machine. #3 should be running on Tx2.

Turn on the fan before you run #3 on Tx2.

# Resource allocation
The funtion "update_allocation" in "urb-OFDMA.py" gives the rules of GPU resouce allocation.

"initial_cfg" stores all application configurations (pre-configured) for devices.

For example, you can change ""fps" to conduct task partition as the total number of tasks is "total_fps". That is, "fps" indicates the number of tasks that are going to be executed by the server and the rest ("total_fps" - "fps") are scheduled as device computation.


# Data
Download dataset: http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images

You need to put the images and labels into the "datasets" folder. So paths follow:

"datasets/images/val2017"

"datasets/labels/val2018"

# Model
weights/download_weights.sh

We need to use "yolov5s", "yolov5m", "yolov5l", "yolov5x". 

Put these weights under the root path (e.g., app/yolov5s).

# Links
1. https://github.com/ultralytics/yolov5
   - Try to read https://github.com/ultralytics/yolov5/blob/master/detect.py
2. https://developer.nvidia.com/embedded/jetson-tx2
