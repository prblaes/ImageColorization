#!/usr/bin/env python

import cv, cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools
from mpi4py import MPI
from glob import glob
import os

from colorizer import Colorizer


def get_grayscale_from_color(color_file):
    '''
    Takes the path to a RGB image file and returns a numpy array of its luminance
    '''
    L, _, _ = cv2.split(cv2.cvtColor(cv2.imread(color_file), cv.CV_BGR2Lab))
    return L


if __name__ == '__main__':

    #change these to point to your training file(s).  Assume that the "images" directory is a symlink to the 
    #cs_229_project Dropbox foler that Rasoul shared


    training_files = ['/shared/users/prblaes/ImageColorization/images/houses/calhouse_0007.jpg']

    input_files = glob('/shared/users/prblaes/ImageColorization/images/houses/*.jpg')

    output_dir = '/shared/users/prblaes/ImageColorization/output/houses/'

    comm = MPI.COMM_WORLD

    #def __init__(self, ncolors=16, probability=False, npca=30, svmgamma=0.1, svmC=1, graphcut_lambda=1):
    ncolors = [16] 
    npca = [32]
    svmgamma = [0.25]
    svmC = [0.5]
    graphcut_lambda = [1]

    params = list(itertools.product(ncolors, npca, svmgamma, svmC, graphcut_lambda, input_files))
    p = params[0]
    
    #which parameter range this node should use
    #chunk_size = int(len(input_files)/comm.size)
    #start = comm.rank * chunk_size
    #stop = start + chunk_size
    c = Colorizer(ncolors=p[0], npca=p[1], svmgamma=p[2], svmC=p[3], graphcut_lambda=p[4])

    #train the classifiers
    c.train(training_files)


    try:

        print('f')


        #for now, convert an already RGB image to grayscale for our input
        grayscale_image = get_grayscale_from_color(input_files[comm.rank])

        #colorize the input image
        colorized_image, g = c.colorize(grayscale_image,skip=2)

        #save the outputs
        #cv2.imwrite('output_gray.jpg', grayscale_image)
        #cv2.imwrite('output_color.jpg', cv2.cvtColor(colorized_image, cv.CV_RGB2BGR))


        l, a, b = cv2.split(cv2.cvtColor(colorized_image, cv.CV_RGB2Lab))
        newColorMap = cv2.cvtColor(cv2.merge((128*np.uint8(np.ones(np.shape(l))),a,b)), cv.CV_Lab2BGR)
        
        cv2.imwrite(output_dir+os.path.basename(input_files[comm.rank]), cv2.cvtColor(colorized_image, cv.CV_RGB2BGR))
        cv2.imwrite(output_dir+'cmap_'+os.path.basename(input_files[comm.rank]), newColorMap)

    except Exception:
        print('\terror: %d'%(comm.rank))

