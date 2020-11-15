### Yolo v1 (2015)

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

### Yolo v2 (2016)
- [YOLOv2: YOLO9000: Better, Faster, Stronger (2016)](https://arxiv.org/abs/1612.08242)
- [Yolo 2 Explained](https://towardsdatascience.com/yolo2-walkthrough-with-examples-e40452ca265f)
- [YOLO v2 implementation in Tensorflow v1](https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html)
- [YOLO v2 implementation in Tensorflow v2](https://www.maskaravivek.com/post/yolov2/) `based on the v1 blog post`
- [experiencor/keras-yolo2](https://github.com/experiencor/keras-yolo2)
- [FairyOnIce/ObjectDetectionYolo](https://github.com/FairyOnIce/ObjectDetectionYolo)

**What's new in Yolo v2**
- Adding Batch Normalization on all of the conv layers in YOLO (which improved mAP by 2%)
- High resolution classifier (original YOLO classifier network was trained on 224x224, in v2 it is increased to 448 for detection)
- Convolutional with anchor boxes (multi-object prediction per grid cell) and also the anchor boxes are defined using K-mean clustering on the training set bounding boxes
- New classification model *Darknet-19* is used as backbone. It is called *Darknet-19* because it has 19 convolutional layers.
- YOLO9000, they mixed the images from both detection (COCO dataset) and classification datasets (ImageNet dataset). Using the WordNet, they merged COCO and ImageNet dataset to form WordTree dataset.

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

### Yolo v3 (2018)
- [YOLOv3: An Incremental Improvement (2018)](https://arxiv.org/abs/1804.02767v1)
- [YOLO, YOLOv2 and YOLOv3: All You want to know](https://medium.com/@amrokamal_47691/yolo-yolov2-and-yolov3-all-you-want-to-know-7e3e92dc4899) `this post explains really well about YOLO basics like loss functions`
- [What’s new in YOLO v3?](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

**What's new in Yolo v3**
- Like Yolo v2, v3 also predicts `tx`, `ty`, `tw`, `th` for each bounding box. In addition to that, it also predicts objectness score (confidence) `c` for each bounding box using *logistic regression*. The value is 1 if the bounding box prior overlaps a ground truth object by more than any other bounding box prior.
- Small objects detection
  - Older YOLO(s) struggle with small objects but in v3 they used shortcut connections (inspired from ResNet) to improve performance on small objects.
  - It allows to get more finer-grained information from the earlier feature map (before the later layers downsampled the input).
  - However compared to previous version, YOLO v3 has worse performance on medium and large size objects.
- New feature extractor network aka *Darknet-53*
  - v2 uses *Darknet-19* but in v3 they use deeper network called *Darknet-53* which is the hybrid approach between *Darknet-19* and *ResNet*, residual network.
  - It has 53 convolution layers so they called it *Darknet-53*.
- Changes in LOSS Function
  - The image below is the loss function used in YOLO v2.
  - In v2, the last three terms are the squared errors. However in v3, it is replaced by *cross-entrophy error*.
  - In other words, object confidence and class predictions in YOLO v3 are now predicted through logistic regression.
 ![yolo v1 loss](https://miro.medium.com/max/534/0*u4UXzV2E_opHIkNs.png)
- Multi labels predictions.
  - In some datasets like the Open Image Dataset an object may has multiple labels. For example, an object can be labeled as a woman and as a person.
  - In v2, using softmax for class prediction imposes the assumption that each box has exactly one class which is often not the case.
  - So in v3, instead of softmax, it simply uses independent logistic classifier for any class. During training, they used *binary cross-entrophy loss* for the class predictions.
  - Using independent logistic classifier, an object can be detected as a woman and as a person at the same time.
- Predictions Across Scales
  - Unlike previous versions which predict output at the last layer, v3 predicts boxes at 3 different scales as below. Image is from [What’s new in YOLO v3? by Ayoosh Kathuria](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b).
  - In order to make predictions at 3 different scales, it also has to upsample the input.
  - First `13x13` layer is to detect large objects, second `26x26` layer is to detect medium objects and the last `52x52` layer is to detect small objects.
  - Because YOLO v3 predicts boxes at 3 different scale, for the same image `416x416` with 3 anchor boxes and stride of 32, YOLO v2 only need to predict 845 (13x13x3) but YOLO v3 predicts 10,647 (13x13x9, 3 anchor boxes with 3 different scales).
  - That is also the reason why YOLO v3 is slower than YOLO v2.
  
  ![darknet-53](https://miro.medium.com/max/1000/1*d4Eg17IVJ0L41e7CTWLLSg.png)
 

<br/>

### Yolo v4 (2020)
Paper related to v4
- [YOLOv4: Optimal Speed and Accuracy of Object Detection (2020)](https://arxiv.org/abs/2004.10934v1)
- [CSPNet: A New Backbone that can Enhance Learning Capability of CNN (2019)](https://arxiv.org/abs/1911.11929)
- [DC-SPP-YOLO: Dense Connection and Spatial Pyramid Pooling Based YOLO for Object Detection (2019)](https://arxiv.org/abs/1903.08589)
- [Feature Pyramid Networks for Object Detection (2017)](https://arxiv.org/abs/1612.03144)
- [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition (2015)](https://arxiv.org/abs/1406.4729v4)
- [Path Aggregation Network for Instance Segmentation (2018)](https://arxiv.org/abs/1803.01534)
- [CBAM: Convolutional Block Attention Module (2018)](https://arxiv.org/abs/1807.06521v2) `for Spatial Attention Module (SAM)`
- [EfficientDet: Scalable and Efficient Object Detection (2019)](https://arxiv.org/abs/1911.09070)`about BiFPN, bi-directional Feature Pyramid Network`

Blog Post and Links
- [RobotEdh/Yolov-4](https://github.com/RobotEdh/Yolov-4/)
- [Yolo v4 Summary](https://jonathan-hui.medium.com/yolov4-c9901eaa8e61)
- [Feature Pyramid Network (Object Detection)](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610)
- [Understanding Feature Pyramid Networks for object detection (FPN)](https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)
- [Review: Spatial Pyramid Pooling](https://medium.com/analytics-vidhya/review-spatial-pyramid-pooling-1406-4729-bfc142988dd2)
- [How does the spatial pyramid pooling method work?](https://www.quora.com/How-does-the-spatial-pyramid-matching-method-work)
- [Reading: PANet — Path Aggregation Network](https://becominghuman.ai/reading-panet-path-aggregation-network-1st-place-in-coco-2017-challenge-instance-segmentation-fe4c985cad1b)
- [Improving instance segmentation using Path Aggregation Network](https://medium.com/analytics-vidhya/improving-instance-segmentation-using-path-aggregation-network-a89588f3d630)
- [Swish Vs Mish: Latest Activation Functions](https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/)
- [Overview on DropBlock Regularization](https://medium.com/swlh/overview-on-dropblock-regularization-b1b9590fd3c2)
- [EfficientDet: Scalable and Efficient Object Detection](https://medium.com/@nainaakash012/efficientdet-scalable-and-efficient-object-detection-ea05ccd28427) `about biFPN`

<br/>

### Some other notable papers
- [YOLO-LITE: A Real-Time Object Detection Algorithm Optimized for Non-GPU Computers (2018)](https://arxiv.org/abs/1811.05588)
