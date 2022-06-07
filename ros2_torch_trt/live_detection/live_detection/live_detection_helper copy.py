'''Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'''


# ROS2 imports 
import rclpy
from rclpy.node import Node

# CV Bridge and message imports
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesisWithPose, BoundingBox2D, Detection2D, Detection2DArray
from cv_bridge import CvBridge, CvBridgeError

from live_detection.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from live_detection.misc import Timer

import cv2
import numpy as np
import os

import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from live_detection.backbone import EfficientDetBackbone

from live_detection.efficientdet.utils import BBoxTransform, ClipBoxes
from live_detection.utils.utils import preprocess_vilab, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import pdb
import glob
from tqdm import tqdm
import random

force_input_size = None
compound_coef = 0
obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']
use_float16 = False
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
threshold = 0.2
iou_threshold = 0.1


class DetectionNode(Node):

    def __init__(self):
        super().__init__('detection_node')

        # Create a subscriber to the Image topic
        # self.subscription = self.create_subscription(Image, 'image', self.listener_callback, 10)
        self.subscription = self.create_subscription(Image, '/usv/slot2/image_raw', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        # Create a Detection 2D array topic to publish results on
        self.detection_publisher = self.create_publisher(Detection2DArray, 'detection', 10)

        # Create an Image publisher for the results
        self.result_publisher = self.create_publisher(Image,'detection_image',10)

        self.net_type = 'mb1-ssd'
        
        # Weights and labels locations
        # self.model_path = os.getenv("HOME")+ '/ros2_models/mobilenet-v1-ssd-mp-0_675.pth'
        # self.label_path = os.getenv("HOME") + '/ros2_models/voc-model-labels.txt'
        
        ## efficientdet
        self.model_path = '/home/vilab/Yet-Another-EfficientDet-Pytorch2/logs/SMD_nohaze/efficientdet-d0_0_10.pth'
        self.obj_list = obj_list



        # self.class_names = [name.strip() for name in open(self.label_path).readlines()]
        # self.num_classes = len(self.class_names)
        
        # self.net = create_mobilenetv1_ssd(len(self.class_names), is_test=True)
        # self.net.load(self.model_path)
        # self.predictor = create_mobilenetv1_ssd_predictor(self.net, candidate_size=200)

        self.use_cuda = True
        self.model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                    ratios=anchor_ratios, scales=anchor_scales)
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.requires_grad_(False)
        self.model.eval()

        if self.use_cuda:
            self.model = self.model.cuda()
        if use_float16:
            self.model = self.model.half()
        self.timer = Timer()
        

    def listener_callback(self, data):
        self.get_logger().info("Received an image! ")
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)

        
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.timer.start()
        # boxes, labels, probs = self.predictor.predict(image, 10, 0.4)
        
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
        
        
        ori_imgs, framed_imgs, framed_metas = preprocess_vilab(image, max_size=input_size)

        if self.use_cuda:
            x = torch.from_numpy(framed_imgs).unsqueeze(0).cuda()
        else:
            x = torch.from_numpy(framed_imgs).unsqueeze(0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        
        
        

        

        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

        out = invert_affine(framed_metas, out)
        
              
        
        # print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        

        detection_array = Detection2DArray()
        
        obj_t = 0
        score_t = 0
        x1_t =0
        y1_t = 0
        x2_t =0
        y2_t = 0

        for j in range(len(out[0]['rois'])):
            x1, y1, x2, y2 = out[0]['rois'][j].astype(np.int)
            obj = obj_list[out[0]['class_ids'][j]]
            score = float(out[0]['scores'][j])
            if score_t < score:
                x1_t, y1_t, x2_t, y2_t = x1, y1, x2, y2
                obj_t = obj
                score_t = score
        interval = self.timer.end()
        print('Time: {:.2f}s, Detect Objects: {:.2f}.'.format(interval, score_t))

        


        bounding_box = BoundingBox2D()
        bounding_box.center.position.x = float((x1_t + x2_t)/2)
        bounding_box.center.position.y = float((y1_t + y2_t)/2)
        bounding_box.center.theta = 0.0
        
        bounding_box.size_x = float(2*(bounding_box.center.position.x  - x1_t))
        bounding_box.size_y = float(2*(bounding_box.center.position.y  - y1_t))



        detection = Detection2D()
        detection.header = data.header
        detection.bbox = bounding_box

        detection_array.header = data.header
        detection_array.detections.append(detection)
        self.detection_publisher.publish(detection_array)

        # Publishing the results onto the the Detection2DArray vision_msgs format
        
        ros_image = self.bridge.cv2_to_imgmsg(cv_image)
        ros_image.header.frame_id = 'camera_frame'
        self.result_publisher.publish(ros_image)

        cv2.rectangle(cv_image, (int(x1_t), int(y1_t)), (int(x2_t), int(y2_t)), (255, 255, 0), 1)
        cv2.putText(cv_image, str(score_t),
                    (x1_t+20, y1_t+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # font scale
                    (255, 0, 255), 2)  # line type

        # # Displaying the predictions
        cv2.imshow('object_detection', cv_image)
        
        cv2.waitKey(1)

      
        # Displaying the predictions

        # for i in range(boxes.size(0)):
        #     box = boxes[i, :]
        #     label = f"{self.class_names[labels[i]]}: {probs[i]:.2f}"
        #     print("Object: " + str(i) + " " + label)
        #     cv2.rectangle(cv_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)

        #     # Definition of 2D array message and ading all object stored in it.
        #     object_hypothesis_with_pose = ObjectHypothesisWithPose()
        #     object_hypothesis_with_pose.id = str(self.class_names[labels[i]])
        #     object_hypothesis_with_pose.score = float(probs[i])

        #     bounding_box = BoundingBox2D()
        #     bounding_box.center.x = float((box[0] + box[2])/2)
        #     bounding_box.center.y = float((box[1] + box[3])/2)
        #     bounding_box.center.theta = 0.0
            
        #     bounding_box.size_x = float(2*(bounding_box.center.x - box[0]))
        #     bounding_box.size_y = float(2*(bounding_box.center.y - box[1]))

        #     detection = Detection2D()
        #     detection.header = data.header
        #     detection.results.append(object_hypothesis_with_pose)
        #     detection.bbox = bounding_box

        #     detection_array.header = data.header
        #     detection_array.detections.append(detection)


        #     cv2.putText(cv_image, label,
        #                (box[0]+20, box[1]+40),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1,  # font scale
        #                (255, 0, 255), 2)  # line type
        # # # Displaying the predictions
        # cv2.imshow('object_detection', cv_image)
        # # Publishing the results onto the the Detection2DArray vision_msgs format
        # self.detection_publisher.publish(detection_array)
        # ros_image = self.bridge.cv2_to_imgmsg(cv_image)
        # ros_image.header.frame_id = 'camera_frame'
        # self.result_publisher.publish(ros_image)
        # cv2.waitKey(1)
        


