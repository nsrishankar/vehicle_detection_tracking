# Colorspace Conversions and analysis functions

import numpy as np
import cv2
import time
import pickle

from collections import deque
from camera_image_calibration import undistort
from tracking_utility import detect_cars,add_heat,apply_heat_threshold,draw_boxes,draw_labeled_boxes, slide_window,ROI
from scipy.ndimage.measurements import label

import classifiers
import feature_analysis
#import neural_networks

# Load saved pickled data for matrix and distortion coefficients
pickle_path='undistortion_pickle.p'

with open(pickle_path,mode='rb') as f:
        undistort_data=pickle.load(f)
undistort_camera_matrix,undistort_coefficients=undistort_data['dist_matrix'], undistort_data['dist_coefficients']

# Video Pipeline
class Tracking_pipeline:
    def __init__(self,classifier,X_scaler):
        self.X_scaler=X_scaler
        self.classifier=classifier
        self.c_space='YCrCb'
        self.orient=9
        self.pix_per_cell=8
        self.cell_per_block=2
        self.spatial_size=(16,16)
        self.nbins=32
        self.threshold=7.5
        #self.scale=(1.5,1.5,1.0)
        #self.y_coord=[[450,650],[400,700],[400,450]]

        self.smoothed_heatmap=deque(maxlen=10)
    
    def avg_heatmap(self,smoothed_heatmap,threshold):
        average_heat_thresholding=0
        for i in range(len(smoothed_heatmap)):
            average_heat_thresholding+=smoothed_heatmap[i]
       
        return apply_heat_threshold(average_heat_thresholding,threshold)

    def tracking(self,raw_image,undistort_matrix=undistort_camera_matrix,undistort_coefficients=undistort_coefficients):
        hog_windows=[]
        
        undistort_image=undistort(raw_image,undistort_matrix,undistort_coefficients) # Undistort images
        copy=np.copy(undistort_image)
        vertices=np.asarray([[[0,400],[raw_image.shape[1],400],[raw_image.shape[1],raw_image.shape[0]],
                          [0,raw_image.shape[0]]]],dtype=np.int32)
        
        roi_image=ROI(copy,vertices) # Region of Interest+Vertices
        
        windows=slide_window(roi_image, x_start_stop=[None,None], y_start_stop=[350,720], 
                    xy_window=(96,96), xy_overlap=(0.5, 0.5)) # Sliding window search
            
        
        bboxes=detect_cars(raw_image=roi_image,window_list=windows,
                           classifier=self.classifier,X_scaler=self.X_scaler,c_space=self.c_space,
                           orient=self.orient,pix_per_cell=self.pix_per_cell,cell_per_block=self.cell_per_block,
                            spatial_size=self.spatial_size,hist_bins=self.nbins)    
                   
        heatmap=np.zeros_like(undistort_image[:,:,0]).astype(np.float)
        
        self.smoothed_heatmap.append(add_heat(heatmap,bboxes))  
        average_heat_thresholding=self.avg_heatmap(self.smoothed_heatmap,self.threshold)
            
        labels=label(average_heat_thresholding) #scipy.ndimage.measurements
        
        return draw_labeled_boxes(undistort_image,labels)

def image_pipeline(raw_image,X_scaler,classifier,
                   undistort_matrix=undistort_camera_matrix,undistort_coefficients=undistort_coefficients):
    
    c_space='YCrCb'
    orient=9
    pix_per_cell=8
    cell_per_block=2
    spatial_size=(16,16)
    nbins=32
    threshold=0.8
    
    undistort_image=undistort(raw_image,undistort_matrix,undistort_coefficients) # Undistort images
    copy=np.copy(undistort_image)
    vertices=np.asarray([[[0,400],[raw_image.shape[1],400],[raw_image.shape[1],raw_image.shape[0]],
                          [0,raw_image.shape[0]]]],dtype=np.int32)
        
    roi_image=ROI(copy,vertices) # Region of Interest+Vertices
        
    windows=slide_window(roi_image, x_start_stop=[None,None], y_start_stop=[350,720], 
                    xy_window=(96,96), xy_overlap=(0.5, 0.5)) # Sliding window search
            
        
    bboxes=detect_cars(raw_image=roi_image,window_list=windows,
                           classifier=classifier,X_scaler=X_scaler,c_space=c_space,
                           orient=orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,
                            spatial_size=spatial_size,hist_bins=nbins)    
                   
    heatmap=np.zeros_like(undistort_image[:,:,0]).astype(np.float)
        
    heatmap=add_heat(heatmap,bboxes)
    heat_thresholding=apply_heat_threshold(heatmap,threshold)
            
    labels=label(heat_thresholding) #scipy.ndimage.measurements
        
    return draw_labeled_boxes(undistort_image,labels)