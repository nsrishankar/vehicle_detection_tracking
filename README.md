# Vehicle Detection & Tracking Pipeline

The goal was to detect and track vehicles in a videostream using camera input. The first step was to callibrate the camera, and correct for distortion. To detect cars vs non-car objects a dataset of car images (various types/positions/colors) and non-car images had features extracted by spatial binning and obtaining histogram of color data(from a colorspace that maximizes difference between cars and roads).

To represent structural data, HOG features were extracted. This in turn, gives a new dataset of approximately 6000 features. Scikit-learn's standard scalar was used to normalized the features, following which, PCA was used for dimensionality reduction.

Various classifiers were tried out: Support Vector Machines, Boosting algorithms, and Ensembles (created from LeNet architectures and KerasClassifier's Voting Classifier).
