import sklearn
import glob
import pickle
import os
import cv2
import numpy as np
import skimage
from skimage import data, color, exposure
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage.measurements import label
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, Flatten, Input, Conv2D, MaxPooling2D, Lambda
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from skimage.transform import resize

from model import *

ymin = 400
ymax = 660
xmax = 1280
box_size = 8

# Create model with bigger input and without final flatten layer
heatmodel = create_model((ymax - ymin, xmax, 3))
heatmodel.load_weights('./weights.13-0.02.hdf5')


def draw_boxes(img, bounding_boxes, color=(0, 0, 255), thick=6):
  draw_img = np.copy(img)
  for bbox in bounding_boxes:
      # Draw a rectangle given bbox coordinates
      cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
  # Return the image copy with boxes drawn
  return draw_img


def search_cars(img):
  cropped = img[ymin:ymax, 0:xmax]
  heat = heatmodel.predict(cropped.reshape(1, cropped.shape[0],
                                           cropped.shape[1], cropped.shape[2]))
  mesh_x, mesh_y = np.meshgrid(np.arange(heat.shape[2]), np.arange(heat.shape[1]))
  x_indices = mesh_x[heat[0,:,:,0]>0.9999999]
  y_indices = mesh_y[heat[0,:,:,0]>0.9999999]
  hot_windows = []
  for x_index, y_index in zip(x_indices,y_indices):
      x_start = x_index * box_size
      y_start = ymin + y_index * box_size
      hot_windows.append(((x_start, y_start),
                          (x_start + box_size * box_size,
                           y_start + box_size * box_size)))
  return hot_windows


def add_heat(heatmap, bounding_boxes):
    for box in bounding_boxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, boxes):
    for car_number in range(1, boxes[1]+1):
        nonzero = (boxes[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    return img


def find_boxes(img):
  hot_windows = search_cars(img)
  heat = np.zeros_like(img[:,:,0]).astype(np.float)
  heat = add_heat(heat,hot_windows)
  heat = apply_threshold(heat, 3)
  heatmap = np.clip(heat, 0, 255)
  boxes = label(heatmap)
  return boxes


def process_image(img):
  boxes = find_boxes(img)
  draw_img = draw_labeled_bboxes(img, boxes)
  return draw_img

for image in glob.glob("test_images/*.jpg"):
    prev_frames = []
    prev_curvatures = []
    prev_car_off = []
    img = skimage.io.imread(image)
    img_lane = process_image(img)
    fig = plt.figure(figsize=(12,20))
    plt.imshow(img_lane)
    fig.savefig('output_images/' + image, bbox_inches='tight', pad_inches=0)

history = None

def pipeline(img):
    boxes = find_boxes(img)
    for car_number in range(1, boxes[1]+1):
        nonzero = (boxes[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        history.append([np.min(nonzerox),np.min(nonzeroy),np.max(nonzerox),np.max(nonzeroy)])
    recent_boxes = np.array(history).tolist()
    boxes = cv2.groupRectangles(recent_boxes, 10, .1)
    if len(boxes[0]) != 0:
        for box in boxes[0]:
            cv2.rectangle(img, (box[0], box[1]), (box[2],box[3]), (0,255,0), 6)
    return img

prev_frames = []
prev_curvatures = []
prev_car_off = []
history = deque(maxlen=30)
video = 'test_video.mp4'
clip = VideoFileClip(video)
clip_output = 'output_videos/' + video
clip_process = clip.fl_image(pipeline)
clip_process.write_videofile(clip_output, audio=False)
