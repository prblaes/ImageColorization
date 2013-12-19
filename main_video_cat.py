#!/usr/bin/env python

import cv, cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools
from mpi4py import MPI
from glob import glob
import os
from datetime import datetime

from colorizer import Colorizer


def get_grayscale_from_color(color_file):
    '''
    Takes the path to a RGB image file and returns a numpy array of its luminance
    '''
    L, _, _ = cv2.split(cv2.cvtColor(cv2.imread(color_file), cv.CV_BGR2Lab))
    return L

def grab_frame(cap):
    frame = cv.QueryFrame(cap)
    img = np.asarray(frame[:,:])
    m,n,_ = img.shape

    img = img[0:int(3*m/4), 0:int(3*n/4)]

    m,n,_ = img.shape
    img = cv2.resize(img, (n/3, m/3))
    
    return img
    

NFRAMES = 100

if __name__ == '__main__':

    #change these to point to your training file(s).  Assume that the "images" directory is a symlink to the 
    #cs_229_project Dropbox foler that Rasoul shared

    comm = MPI.COMM_WORLD
    
    ##grab frames from video
    #if comm.rank == 0:
    #    cap = cv.CaptureFromFile('images/cat.mp4')
    #    
    #    #skip ahead
    #    i=0
    #    while i < 150:
    #        cv.QueryFrame(cap)
    #        i+=1

    #    
    #    training_file = grab_frame(cap)
    #    imwrite('/shared/users/prblaes/ImageColorization/tmp/training.png', training_file)

    #    i=0
    #    while i < NFRAMES:
    #        f = grab_frame(cap)
    #        cv2.imwrite('/shared/users/prblaes/ImageColorization/tmp/frame%d.png'%i, f)
    #        i+=1

    #
    ##syncrhonize here
    #comm.Barrier()

    training_files = ['/shared/users/prblaes/ImageColorization/tmp/training.png']

    input_files = glob('/shared/users/prblaes/ImageColorization/tmp/frame*.png')

    output_dir = '/shared/users/prblaes/ImageColorization/video_%s/'%(datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'))
    
    if comm.rank == 0:
        os.mkdir(output_dir)

    #comm.Barrier()


    #def __init__(self, ncolors=16, probability=False, npca=30, svmgamma=0.1, svmC=1, graphcut_lambda=1):
    ncolors = [8] 
    npca = [64]
    svmgamma = [0.1]
    svmC = [1]
    graphcut_lambda = [1.1]

    params = list(itertools.product(ncolors, npca, svmgamma, svmC, graphcut_lambda, input_files))
    p = params[0]
    
    #which parameter range this node should use
    chunk_size = int(len(input_files)/comm.size)
    start = comm.rank * chunk_size
    stop = start + chunk_size

    c = Colorizer(ncolors=p[0], npca=p[1], svmgamma=p[2], svmC=p[3], graphcut_lambda=p[4])

    #train the classifiers
    c.train(training_files)
    
    try:
        for file in input_files[start:stop]:
            print('Processing: '+os.path.basename(file))

            #for now, convert an already RGB image to grayscale for our input
            grayscale_image = get_grayscale_from_color(file)

            #colorize the input image
            colorized_image, g = c.colorize(grayscale_image,skip=2)

            l,a,b = cv2.split(cv2.cvtColor(colorized_image, cv.CV_BGR2Lab))
            
            a_new = cv2.medianBlur(a, 15)
            b_new = cv2.medianBlur(b, 15)

            img_new = cv2.cvtColor(cv2.merge((l, a_new, b_new)), cv.CV_Lab2BGR)

            #l, a, b = cv2.split(cv2.cvtColor(colorized_image, cv.CV_RGB2Lab))
            #newColorMap = cv2.cvtColor(cv2.merge((128*np.uint8(np.ones(np.shape(l))),a,b)), cv.CV_Lab2BGR)
            
            cv2.imwrite(output_dir+'colorized_'+os.path.basename(file), cv2.cvtColor(img_new, cv.CV_RGB2BGR))

    except Exception, e:
        print('\terror: %d\n%s'%(comm.rank, e))

