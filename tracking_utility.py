# Functions to detect and track vehicles using image structure/color, heatmap, heatmap-threshold and bounding boxes

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import pickle
import cv2

from feature_analysis import display,colorspace_conversion,hog_structure_features,color_histogram,spatial_bins,feature_extraction, single_img_features
from scipy.ndimage.measurements import label

# Functions using hog sub-sampling to make predictions of found cars within a certain border
# Using all HOG channels
def detect_cars(raw_image,window_list,classifier,X_scaler,c_space,
                orient,pix_per_cell,cell_per_block,spatial_size,hist_bins):
    bounding_boxes=[]
    #raw_image=raw_image.astype(np.float32)/255
    for window in window_list:
        search_image=cv2.resize(raw_image[window[0][1]:window[1][1],window[0][0]:window[1][0]],(64,64))
    
        features_extracted=single_img_features(search_image, c_space='YCrCb', 
                                               spatial_size=(32, 32),hist_bins=32, 
                                               orient=9,pix_per_cell=8, cell_per_block=2,
                                               hog_channel='ALL',use_spatialbins=True, use_histogram=True,use_hog=True)
        features_extracted=np.array(features_extracted).reshape(1,-1)
        test_features=X_scaler.transform(features_extracted)
        test_prediction=classifier.predict(test_features)

        if test_prediction==1:
            bounding_boxes.append(window)
                
    return bounding_boxes

# Creating a sliding window to fit cars
def slide_window(raw_image, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = raw_image.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = raw_image.shape[0]
    # Compute the span of the region to be searched    
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]
    
    # Compute the number of pixels per step in x/y
    x_pix_per_step=np.int(xy_window[0]*(1-xy_overlap[0]))
    y_pix_per_step=np.int(xy_window[1]*(1-xy_overlap[1]))
    # Compute the number of windows in x/y
    x_buffer=np.int(xy_window[0]*xy_overlap[0])
    y_buffer=np.int(xy_window[1]*xy_overlap[1])
    x_windows=np.int((x_span-x_buffer)/x_pix_per_step)
    y_windows=np.int((y_span-y_buffer)/y_pix_per_step)
    
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for y_window in range(y_windows):
        for x_window in range(x_windows):
            x_start=x_start_stop[0]+x_window*x_pix_per_step
            x_end=x_start+xy_window[0]
            y_start=y_start_stop[0]+y_window*y_pix_per_step
            y_end=y_start+xy_window[1]
            window_list.append(((x_start,y_start),(x_end,y_end)))
    # Return the list of windows
    return window_list

# Narrowing search to specific region of interest (P1)
def ROI(raw_image,vertices):
    mask = np.zeros_like(raw_image)
    h,w,ch=raw_image.shape
    if len(raw_image.shape) > 2:
        ignore_mask_color=(255,)*ch
    else:
        ignore_mask_color=255

    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_image = cv2.bitwise_and(raw_image,mask)
    return masked_image

# Increase the heat of successful detections
def add_heat(init_heatmap,bboxes_list):
    for bbox in bboxes_list:
        init_heatmap[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]+=1
    return init_heatmap

# Set heatmaps below a certain threshold to be null (prevents false positives or multiple detections)
def apply_heat_threshold(init_heatmap,threshold):
    init_heatmap[init_heatmap<=threshold]=0
    return init_heatmap

# Draw a bounding box on a raw image
def draw_boxes(raw_image,bboxes_list,color=(0,0,255),thickness=6):
    copy=np.copy(raw_image)
    
    for bbox in bboxes_list:
        cv2.rectangle(copy,bbox[0],bbox[1],color,thickness)
    return copy

def draw_labeled_boxes(raw_image,labels):
    h,w,_=raw_image.shape
    mid_x=w/2
    
    y_px2m=10/50
    x_px2m=3.7/750
    for detect_car in range(1,labels[1]+1): # For all detected cars
        nonzero=(labels[0]==detect_car).nonzero() # Find pixels with each car number label values
        
        nonzero_y=np.array(nonzero[0])
        nonzero_x=np.array(nonzero[1])
        
        bbox=((np.min(nonzero_x),np.min(nonzero_y)),(np.max(nonzero_x),np.max(nonzero_y))) # Bounding box
        
        bbox_x=int(0.5*(bbox[0][0]+bbox[1][0]))
        bbox_y=int(1.05*(bbox[0][1]))
        
        dist_m=np.sqrt((x_px2m*(np.min(nonzero_x)-mid_x))**2+(y_px2m*(np.max(nonzero_y)-h))**2)
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        text='{} m'.format(round(dist_m,3))
        cv2.rectangle(raw_image,bbox[0],bbox[1],(0,0,255),6)
        cv2.putText(raw_image,text,(bbox_x,bbox_y),font,0.5,(0,0,255),1,cv2.LINE_AA)
    return raw_image