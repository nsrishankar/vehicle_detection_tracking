# Vehicle Detection & Tracking Pipeline
![alt_text](/sample_outputs/drive_sample_single.gif)

![alt_text](/sample_outputs/drive_sample_multiple.gif)

![alt_text](/sample_outputs/night_sample.gif)

Full Video: [YouTube](https://www.youtube.com/watch?v=Bvbh35YDcnk&index=11&list=PLMr_u-BsTKSoWrumKl-4sDf_keQxDZFaa)


The goal was to detect and track vehicles in a videostream using camera input. The first step was to callibrate the camera, and correct for distortion. To detect cars vs non-car objects a dataset of car images (various types/positions/colors) and non-car images had features extracted by spatial binning and obtaining histogram of color data(from a colorspace that maximizes difference between cars and roads).

To represent structural data, HOG features were extracted. This in turn, gives a new dataset of approximately 6000 features. Scikit-learn's standard scalar was used to normalized the features, following which, PCA was used for dimensionality reduction.

![alt_text](/Images/car-and-hog.jpg)

Various classifiers were tried out: Support Vector Machines, Boosting algorithms, and Ensembles (created from LeNet architectures and KerasClassifier's Voting Classifier).

For the tracking functionality, a sliding window algorithm with various size windows was implemented, which searched a region-of-interest in the image (lower half of image, with most likelihood of having cars). To avoid false positives, and combine multiple detections, a heatmap is constructed with hot windows to draw a cohesive bounding polytope and smoothed over five subsequent frames.


![alt_text](/Images/Heatmaps.png)

![alt_text](/Images/Thresholded_heatmaps_3.png)

![alt_text](/Images/Bounding_boxes_2.png)


As can be seen this is not perfect- there are still false positves detected especially in an unseen scenario (i.e. night driving).

## Improvements
- Further augment the dataset using car/non-car images in various driving conditions.
- Implement near-real time YOLO or Single-Shot for detection.
