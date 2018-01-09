# Image Calibrations and Viewpoint transformations
# Functions that takes in a set of chessboard images to output a camera matrix and distortion coefficients 

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

# Returns the corners of an image
def calibrate_camera(images_path, pattern=(9,6), draw_chessboard=True):
    print("Camera Calibration initialized.")
    images_path=glob.glob(images_path)
    obj_points=np.zeros((pattern[1]*pattern[0],3),np.float32)
    obj_points[:,:2]=np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)
    
    # Array objects
    obj_array=[] # 3D points in real-world object space
    img_array=[] # 2D points in image space
    
    # Plotting
    fig, ax=plt.subplots(5,4, figsize=(10, 10))
    ax=ax.ravel()
    cal_images=[]
    uncal_images=[]
    for i, path in enumerate(images_path):
        image=cv2.imread(path)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        # Chessboard Corners
        ret, corners=cv2.findChessboardCorners(gray,(pattern[0],pattern[1]),None)
        
        if ret:
            obj_array.append(obj_points)
            img_array.append(corners)
            print("Detected corners for image {}.".format(i))
            cal_images.append(image)
            
        else:
            uncal_images.append(image)
            
        if draw_chessboard:
            # print("Corrected images with detected corners.")
            for image in cal_images:
                image=cv2.drawChessboardCorners(image,pattern,corners,ret)
                ax[i].axis('off')
                ax[i].imshow(image)
    print("(9,6) pattern corners cannot be detected in {} images in the camera_cal folder.".format(len(np.asarray(uncal_images))))
    return obj_array,img_array,cal_images,uncal_images

# Takes in a raw_image, object and image corner/array to undistort the image.
def undistortion(raw_image, object_points, image_points):
    image_shape=(raw_image.shape[0],raw_image.shape[1])
    
    ret,camera_matrix,distortion_coefficients,rvecs,tvecs=cv2.calibrateCamera(object_points, image_points, image_shape,
                                                                              None, None)
    destination_image=cv2.undistort(raw_image, camera_matrix, distortion_coefficients, None, camera_matrix)
    
    return destination_image, camera_matrix, distortion_coefficients

# Define undistortion for any future images and videos after chessboard calibration
def undistort(raw_image, camera_matrix,distortion_coefficients):
    return cv2.undistort(raw_image, camera_matrix, distortion_coefficients, None, camera_matrix)