#! /usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
Code to perform image transformations and dimensionality reduction.
Author: Harsh Bhate
Date: April 15
"""

import cv2
import gym
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class transformImage(object):
    """Class that performs image transformation.
    """
    
    def __init__(self):
        pass
    
    def display_image(self, image):
        """Function to display image"""
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def grayscale(self, image, scale=0.50):
        """Function to standardize image and convert to float
        """
        img = cv2.resize(image,
                        None,
                        fx=scale,
                        fy=scale)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def pca_compression(self, image, component=0.75):
        """Function to perform PCA on the image
        """
        img = normalize(image)
        pca = PCA(component)
        return pca.fit_transform(img)

    def threshold(self, image):
        """Uses average thresholding to binarize image
        """
        if image.ndim > 2:
            raise ValueError('Please send a singular vector')
        if image.ndim == 2:
            _, component = np.shape(image)
            if component != 1:
                raise ValueError('Please check Vector Dimension')
            else:
                image = image.flatten()
        if image.ndim == 0:
            raise ValueError('Please send a valid vector')
        zero_threshold = 0
        mean = np.mean(image)


