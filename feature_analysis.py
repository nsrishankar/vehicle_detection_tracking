# Colorspace Conversions and analysis functions

import numpy as np
import cv2
import time

from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

# Converts an image from BGR-space (read using OpenCV2) to RGB-space (display using Matplotlib).\
def display(raw_image):
    return cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)

# Converts a raw image to another colorspace: YUV, LUV, HLS, HSV, YCrCb, RGB
def colorspace_conversion(raw_image,c_space):
    raw_image=np.copy(raw_image)
    if c_space!='RGB':
        if c_space=='YUV':
            colorspace_image=cv2.cvtColor(raw_image,cv2.COLOR_RGB2YUV)
        elif c_space=='LUV':
            colorspace_image=cv2.cvtColor(raw_image,cv2.COLOR_RGB2LUV)
        elif c_space=='HLS':
            colorspace_image=cv2.cvtColor(raw_image,cv2.COLOR_RGB2HLS)
        elif c_space=='HSV':
            colorspace_image=cv2.cvtColor(raw_image,cv2.COLOR_RGB2HSV)
        elif c_space=='YCrCb':
            colorspace_image=cv2.cvtColor(raw_image,cv2.COLOR_RGB2YCrCb)
    else:
        colorspace_image=raw_image
    return colorspace_image

# Create a color-histogram to get color-based features of images
def color_histogram(raw_image,nbins=32,brange=(0,256)):
    c1_hist,_=np.histogram(raw_image[:,:,0],bins=nbins,range=brange)
    c2_hist,_=np.histogram(raw_image[:,:,1],bins=nbins,range=brange)
    c3_hist,_=np.histogram(raw_image[:,:,2],bins=nbins,range=brange)
    concat=np.concatenate((c1_hist,c2_hist,c3_hist))
    return concat

# Create a spatial-bin to get another color-based feature of images
def spatial_bins(raw_image, resized=(16,16)): 
    return cv2.resize(raw_image,resized).ravel()

# (HOG) Create a histogram-of-oriented-gradients with options of visualizing hog outputs or obtaining a feature vector
def hog_structure_features(raw_image,orient,pix_per_cell,cell_per_block,visualise=False,feature_vec=True):
    if visualise==True:
        features,hog_image=hog(raw_image,orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),
                               cells_per_block=(cell_per_block,cell_per_block),visualise=visualise,
                               feature_vector=feature_vec,transform_sqrt=True)
        return features, hog_image
    else:
        features=hog(raw_image,orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),
                               cells_per_block=(cell_per_block,cell_per_block),visualise=visualise,
                               feature_vector=feature_vec,transform_sqrt=True)
        return features
def feature_extraction(images,c_space,orient,pix_per_cell,cell_per_block,use_histogram=True,use_spatialbins=True,
                       use_hog=True,hog_channel='ALL'):
    features=[]
    for image in images:
        temp_features=[]
        new_colorspace_image=colorspace_conversion(image,c_space)
                                     
        if use_histogram==True:
            # Color Histograms
            color_histograms=color_histogram(new_colorspace_image,nbins=32,brange=(0,256))
            temp_features.append(color_histograms)
        if use_spatialbins==True:
            # Spatial binning of color images
            color_spatial_bins=spatial_bins(new_colorspace_image, resized=(16,16))
            temp_features.append(color_spatial_bins)
        if use_hog==True:
            # Histogram of Oriented Gradients with no visualization and outputs just a feacture vector
            if hog_channel=='ALL':
                hog_features=[]
                for channel in range(new_colorspace_image.shape[2]):
                    hog_feature_channel=hog_structure_features(new_colorspace_image[:,:,channel],
                                                               orient,pix_per_cell,cell_per_block,
                                                              visualise=False,feature_vec=True)
                    hog_features.append(hog_feature_channel)
                hog_features=np.ravel(hog_features)
            else:
                hog_features=np.ravel(hog_structure_features(new_colorspace_image[:,:,hog_channel],
                                                               orient,pix_per_cell,cell_per_block,
                                                              visualise=False,feature_vec=True))
            temp_features.append(hog_features)
        features.append(np.concatenate(temp_features))
    return features

def single_img_features(img, c_space='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                        use_spatialbins=True, use_histogram=True, use_hog=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if c_space != 'RGB':
        if c_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif c_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif c_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif c_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif c_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)   
    new_colorspace_image=feature_image
    if use_histogram==True:
        # Color Histograms
        color_histograms=color_histogram(new_colorspace_image,nbins=32,brange=(0,256))
        img_features.append(color_histograms)
    if use_spatialbins==True:
        # Spatial binning of color images
        color_spatial_bins=spatial_bins(new_colorspace_image, resized=(16,16))
        img_features.append(color_spatial_bins)
    if use_hog==True:
    # Histogram of Oriented Gradients with no visualization and outputs just a feacture vector
        if hog_channel=='ALL':
            hog_features=[]
            for channel in range(new_colorspace_image.shape[2]):
                hog_feature_channel=hog_structure_features(new_colorspace_image[:,:,channel],
                                                            orient,pix_per_cell,cell_per_block,
                                                              visualise=False,feature_vec=True)
                hog_features.extend(hog_feature_channel)
                #hog_features=np.ravel(hog_features)
        else:
            hog_features=np.ravel(hog_structure_features(new_colorspace_image[:,:,hog_channel],
                                                               orient,pix_per_cell,cell_per_block,
                                                              visualise=False,feature_vec=True))
        img_features.append(hog_features)

    return np.concatenate(img_features) 


# Perform dimensionality reduction on features
def perform_pca(dataset,n_components=None,perform_pca=True):
    if perform_pca:
        pca=PCA(n_components=n_components,svd_solver='randomized').fit(dataset)
        reduced_dataset=pca.transform(dataset)
        
        #pcafile='pca.save'
        #pcasave=joblib.dump(pca,pcafile)
        #print("Saved PCA model")
        
        return reduced_dataset,pca
    
    else:
        return dataset

# Find the optimium number of components for dimension reduction using a simple/Linear-SVM classifier
def pca_ocheck(X_dataset,y_dataset):
    t0_pcacheck=time.time()
    pca=PCA(svd_solver='randomized')
    clf=LinearSVC()
    
    pipeline=Pipeline(steps=[('pca',pca),('svm',clf)])
    
    n_components=(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000)
   
    random_estimator=GridSearchCV(pipeline,dict(pca__n_components=n_components))
    random_estimator.fit(X_dataset,y_dataset)
    t1_pcacheck=time.time()
    print("GridSearchCV took {} seconds.".format(round(t1_pcacheck-t0_pcacheck,3)))
    
    return (random_estimator.cv_results_,random_estimator.best_estimator_)