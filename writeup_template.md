# Writeup - Vehicle Detection Project -

For this project, I decided not to implement the Histogram of Oriented Gradients
(HOG) approach for feature extraction but instead train a convolutional neural
network on car images following the suggestions from numerous students in the
`#s-t1-deep-learning` channel.

* Run your pipeline on a video stream (start with the test_video.mp4 and later
  implement on full project_video.mp4) and create a heat map of recurring
  detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/training_data_overview.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Model architecture and training

### Model

The model architecture is a very simple one, consisting almost exclusively of
convolutional layers connected with rectified linear units (`relu`) with only
one pooling layer as inspired by
https://arxiv.org/pdf/1412.6806.pdf and various discussions in the
`#s-t1-deep-learning` channel.

The model definition is found in `model.py`, which is used both for the script
to train the model (`train.py`) and the script to run the model in the image
processing pipeline (`run.py`). To avoid overfitting, I add dropout layers
with 50 % drop out inbetween the convolution layers.
```
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 64, 64, 3)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 128)       3584
_________________________________________________________________
dropout_1 (Dropout)          (None, 64, 64, 128)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 128)       147584
_________________________________________________________________
dropout_2 (Dropout)          (None, 64, 64, 128)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 64, 64, 128)       147584
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 128)         0
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 8, 128)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 1, 128)         1048704
_________________________________________________________________
dropout_4 (Dropout)          (None, 1, 1, 128)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 1, 1)           129
=================================================================
Total params: 1,347,585
Trainable params: 1,347,585
Non-trainable params: 0
```
This is followed for the training by a final `Flatten` layer to be able to
compare with the binary car/non-car label. When running the model in the
pipeline, I omit this layer as I want to use the image output of the
convolutional network.

### Udacity training set
I have used the standard training set of udacity of
[vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
and
[non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
for training the model.  This contains 64x64 images with 3 color channels. With
a 10 % split of validation to training data, I have
```
15984 train samples
1776 test samples
```
whereby the distribution of cars to non-cars is fairly uniform in both data
sets:
```
8101 noncars to 7883 cars in training set (1 : 1.02765444628 )
867 noncars to 909 cars in test set (1 : 0.953795379538 )
```
This is important to avoid that the model prefers one category over the other
in training or validation. Here is an example from the set, whereby I label
cars as `1.0` and non-cars as `0.0`:

![alt text][image1]

### Training results and tuning
As mentioned before, the code to train the model is found in `train.py`.
I have experimented with both `mean_squared_error` and `mean_absolute_error` as
well as `Adam` and `RMSprop` optimizers. I got the best results with the
`mean_squared_error` and the `RMSprop` optimizer. To avoid overfitting, I
activated an `EarlyStopping` callback. To make optimal use of the `g2.2xlarge`
machine, I used a batch size of 128.

The final training run had the following history:

## Combining the results of the neural network
To reuse our model that has been trained on the 64x64 images, we simply change
the input dimensions from `(64,64,3)` to `(ymax - ymin, xmax, 3)`, whereby I
only feed in the part of the image that is relevant for vehicle detection `ymax
= 660` and `ymin = 400`. On the horizontal, the whole image is used, thus `xmax
= 1280`. The cropping in the vertical speeds up the processing and avoids
bogus detections. With this adapted model, we load in the trained model weights
and use it to `predict`. The output of this are images themselves attached with
the probability that it is a car. I only keep images that have a probability
that is higher than `probability_threshold = 0.999`. Convolutional neural
networks seem really appropriate for this task as we don't have to invent a
weird sliding window search but obtain this naturally. This output is shown
here as the small red windows:

![alt text][image2]

All other test images can be found in [output_images/test_images]

The basic pipeline is described in `find_boxes` of `run.py`.

## Sliding Window Search


## Video implementation

## Discussion






### Sliding Window Search

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

