#!/usr/bin/env python

import cv, cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools
from mpi4py import MPI

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


    training_files = ['/shared/users/prblaes/ImageColorization/images/houses/calhouse_0001.jpg' ]
    input_file = '/shared/users/prblaes/ImageColorization/images/houses/calhouse_0007.jpg'

    output_dir = '/shared/users/prblaes/ImageColorization/output/param_sweep/'

    comm = MPI.COMM_WORLD

    #def __init__(self, ncolors=16, probability=False, npca=30, svmgamma=0.1, svmC=1, graphcut_lambda=1):
    ncolors = [16, 32, 64] 
    npca = [16, 32, 64, 128]
    svmgamma = [0.1, 0.25, 0.5, 1, 2]
    svmC = [0.1, 0.25, 0.5, 1, 2, 5]
    graphcut_lambda = [0, 1, 2, 5, 10, 100]

    params = list(itertools.product(ncolors, npca, svmgamma, svmC, graphcut_lambda))
    
    #which parameter range this node should use
    chunk_size = int(len(params)/comm.size)
    start = comm.rank * chunk_size
    stop = start + chunk_size

    for (ind, p) in enumerate(params[start:stop]):

        try:


            print('image: %d\t ncolors=%f, npca=%f, svmgamma=%f, svmC=%f, graphcut_lambda=%f'%(comm.rank*chunk_size+ind, p[0], p[1], p[2], p[3], p[4]))
            c = Colorizer(ncolors=p[0], npca=p[1], svmgamma=p[2], svmC=p[3], graphcut_lambda=p[4])

            #train the classifiers
            c.train(training_files)

            #for now, convert an already RGB image to grayscale for our input
            grayscale_image = get_grayscale_from_color(input_file)

            #colorize the input image
            colorized_image, g = c.colorize(grayscale_image,skip=1)

            #save the outputs
            #cv2.imwrite('output_gray.jpg', grayscale_image)
            #cv2.imwrite('output_color.jpg', cv2.cvtColor(colorized_image, cv.CV_RGB2BGR))


            l, a, b = cv2.split(cv2.cvtColor(colorized_image, cv.CV_RGB2Lab))
            newColorMap = cv2.cvtColor(cv2.merge((128*np.uint8(np.ones(np.shape(l))),a,b)), cv.CV_Lab2BGR)
            
            cv2.imwrite(output_dir+'out_%d.png'%(comm.rank*chunk_size + ind), cv2.cvtColor(colorized_image, cv.CV_RGB2BGR))
            cv2.imwrite(output_dir+'cmap_%d.png'%(comm.rank*chunk_size + ind), newColorMap)

        except Exception:
            print('\terror: %d'%(comm.rank*chunk_size + ind))
 
