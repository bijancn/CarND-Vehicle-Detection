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
[image2]: ./output_images/test_images/test1.jpg
[training1]: ./hist_accuracy.png
[training2]: ./hist_loss.png
[image3]: ./output_images/heat/test_images/test1.jpg

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

The final training run had the following history of validation accuracy and loss

![alt text][training1]
![alt text][training2]

reaching `val_loss: 0.0038 - val_acc: 0.9949` at epoch 12, which did not improve
over the next 5 epochs, leading to an early stop in epoch 17. The finally used
weights for the next section thus correspond to epoch 12.

## Combining the results of the neural network
To reuse our model that has been trained on the 64x64 images, we simply change
the input dimensions from `(64,64,3)` to `(ymax - ymin, xmax, 3)`, whereby I
only feed in the part of the image that is relevant for vehicle detection, i.e.
`ymax = 660` and `ymin = 400`.  Keras then takes care to adapt the dimensions of
all following layers.  On the horizontal, the whole image is used, thus `xmax =
1280`. The cropping in the vertical speeds up the processing and avoids bogus
detections. With this adapted model, we load in the trained model weights and
use it to `predict` in the `search_cars` function. The output of this are images
themselves attached with the probability that it is a car. I only keep images
that have a probability that is higher than `probability_threshold = 0.999999`.
Convolutional neural networks seem really appropriate for this task as we don't
have to come up with a sliding window strategy but obtain this naturally. To be
specific, the output shape of the last layer is `(25, 153, 1)` instead of `(1,
1, 1)`.  This output is shown here as the small red windows:

![alt text][image2]

All other test images can be found in
[output_images/test_images](output_images/test_images). To decide which of
these to keep, we can add for each of these windows +1, see `add_heat`, to
generate a heat map for the whole image. Hereby, we only keep the pixels that
have at least three positive confirmations to reduce the number of false
positives. Finally, we use `scipy.ndimage.measurements.label` to find labelled
features in this heat map. I assume that each of the found blobs correspond to a
vehicle. This constitutes the main logic of the pipeline to `find_boxes`.

Here is an example of the heat map for the same image shown above. All other
heat maps for the test images can be found in
[output_images/heat/test_images](output_images/heat/test_images).

![alt text][image3]

---

## Video implementation
Here's a [link to my video result](./project_video.mp4).

In addition to the logic that was implemented for single pictures as described
in the last section, I keep track of the last 30 frames and use
`cv2.groupRectangles` with a group threshold of 10 and an epsilon 0.1 to cluster
the similar rectangles together.

---

## Discussion


### Video Implementation

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.
From the positive detections I created a heatmap and then thresholded that map
to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()`
to identify individual blobs in the heatmap.  I then assumed each blob
corresponded to a vehicle.  I constructed bounding boxes to cover the area of
each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

## Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
