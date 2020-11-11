### Yolo v1

- [YOLOv1: You Only Look Once: Unified, Real-Time Object Detection (2015)](https://arxiv.org/abs/1506.02640)  
- [Video explanation about YOLO v1](https://youtube.com/watch?v=n9_XyCGr-MI)
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)  
- [Overview of the YOLO Object Detection Algorithm](https://medium.com/@ODSC/overview-of-the-yolo-object-detection-algorithm-7b52a745d3e0)
- [IoU - Intersection over Union (video)](https://youtu.be/XXYG5ZWtjj0)
- [mAP (mean Average Precision) for Object Detection](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
- [mean Average Precision (video)](https://youtu.be/FppOzcDvaDI)
- [Non-maximum Suppression (NMS)](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c) `to fix multiple detections although it is not very effective as mentioned in paper, page 4.`
- [Non-max Suppression (video)](https://youtu.be/YDkjWEN8jNA)
- [YOLO loss function explanation](https://stats.stackexchange.com/a/287497)

<br/>

### Yolo v2
- [YOLOv2: YOLO9000: Better, Faster, Stronger (2016)](https://arxiv.org/abs/1612.08242)
- [Yolo 2 Explained](https://towardsdatascience.com/yolo2-walkthrough-with-examples-e40452ca265f)
- [YOLO v2 implementation in Tensorflow v1](https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html)
- [YOLO v2 implementation in Tensorflow v2](https://www.maskaravivek.com/post/yolov2/) `based on the v1 blog post`
- [experiencor/keras-yolo2](https://github.com/experiencor/keras-yolo2)
- [FairyOnIce/ObjectDetectionYolo](https://github.com/FairyOnIce/ObjectDetectionYolo)

```
YOLO v2 TLDR;

SETUP HELPER METHODS;
- Encode image by resizing the input image and normalize the image
- Decode image by rescale center_x, center_y, center_w, center_h to image original dimension
- Find best anchor box by calculating IoU (Intersection over Union)
- Method to scan over w, h grid of the image and find the object class if the object confidence is 1 (or over some threshold)


SETUP LOSS METHODS;
- Calculate loss between predicted (x, y, w, h) and ground truth (x, y, w, h)
- Calculate loss for object classification (this is simple classification loss func)
- Calculate loss for object confidence, For each (grid cell, and anchor)
  - Calculate loss for IoU
      - Get true box confidence by calculating IoU score between (ground truth and predicted bounding box)
      - For each predicted bounding box, calculate the best IoU regardless of ground truth anchor box
  - Calculate object confidence mask
      - Filter worse IoU with some threshold (for example < 0.6, lesser IoU means worse)
      - And multiply the filtered above with (1 - true box confidence)
      - After that add the result with true box confidence IoU
      - this is to penalize the confidence of the anchor boxes, which are responsible for correspoding grouth truth box
  - (True Box Confidence - Predicted Box Confidence) * object confidence mask -> overall object confidence loss
- All {x, y, w, h} loss + object classification loss + object confidence loss -> Yolo loss


SETUP TRAINING WORKFLOW;
Use k-means clustering on trianing dataset to find anchor boxes  
    Use elbow method to find good k anchor boxes

Setup Darknet-19

Load Yolo v2 weights into Darknet-19; except the last convolution layers for prediction
    Freeze the early conv layers
    The last prediciton layer will be trained

Train the model with custom loss function we defined above

Predicting the image
    Among all predictions, find highest probability bounding box class
    Use non-max suppression to choose the bounding box with highest IoU
```

> The above pseudocode is TLDR; of this [yolov2 notebook](https://github.com/the-robot/deeplearning/blob/master/object-detection/yolo/yolov2.ipynb).

<br/>

### Yolo v3
- [YOLOv3: An Incremental Improvement (2018)](https://arxiv.org/abs/1804.02767v1)
- [YOLO, YOLOv2 and YOLOv3: All You want to know](https://medium.com/@amrokamal_47691/yolo-yolov2-and-yolov3-all-you-want-to-know-7e3e92dc4899) `this post explains really well about YOLO basics like loss functions`
- [What’s new in YOLO v3?](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

<br/>

### Yolo v4
Paper related to v4
- [YOLOv4: Optimal Speed and Accuracy of Object Detection (2020)](https://arxiv.org/abs/2004.10934v1)
- [CSPNet: A New Backbone that can Enhance Learning Capability of CNN (2019)](https://arxiv.org/abs/1911.11929)
- [Feature Pyramid Networks for Object Detection (2017)](https://arxiv.org/abs/1612.03144)

Blog Post and Links
- [RobotEdh/Yolov-4](https://github.com/RobotEdh/Yolov-4/)
- [Yolo v4 Summary](https://jonathan-hui.medium.com/yolov4-c9901eaa8e61)
- [Feature Pyramid Network (Object Detection)](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610)
- [Understanding Feature Pyramid Networks for object detection (FPN)](https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)
- [Swish Vs Mish: Latest Activation Functions](https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/)

<br/>

### Some other notable papers
- [YOLO-LITE: A Real-Time Object Detection Algorithm Optimized for Non-GPU Computers (2018)](https://arxiv.org/abs/1811.05588)
