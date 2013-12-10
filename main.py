#!/usr/bin/env python

import cv, cv2
import matplotlib.pyplot as plt
import numpy as np

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


    training_files = ['images/houses/calhouse_0001.jpg', 'images/houses/calhouse_0002.jpg']
                      
    input_file = 'images/houses/calhouse_0007.jpg'


    #training_files = ['images/book_chapter/islande.jpg' ]
    #input_file = 'images/book_chapter/paysage_gris.png'

    #training_files = ['test/jp.jpg' ]
    #input_file = 'test/chris.jpg'

    #training_files = ['images/houses/calhouse_0001.jpg' ]
    #input_file = 'images/houses/calhouse_0002.jpg'
    
    #training_files = ['test/ch1.jpg']
    #input_file = 'test/ch1.jpg'
    
    #training_files = ['images/cats/cat.jpg','images/cats/cats4.jpg']
    #input_file = 'images/cats/cats3.jpg'
    
    c = Colorizer(probability=False)

    #train the classifiers
    c.train(training_files)

    #for now, convert an already RGB image to grayscale for our input
    grayscale_image = get_grayscale_from_color(input_file)

    #colorize the input image
    colorized_image, g = c.colorize(grayscale_image,skip=8)

    print('min g = %f, max g = %f'%(np.min(g), np.max(g)))

    #save the outputs
    cv2.imwrite('output_gray.jpg', grayscale_image)
    cv2.imwrite('output_color.jpg', cv2.cvtColor(colorized_image, cv.CV_RGB2BGR))

    # prep new color map:
    l, a, b = cv2.split(cv2.cvtColor(colorized_image, cv.CV_RGB2Lab))
    newColorMap = cv2.merge((128*np.uint8(np.ones(np.shape(l))),a,b))

    # compute prediction error:
    l_target, a_target, b_target = cv2.split(cv2.cvtColor(cv2.imread(input_file), cv.CV_BGR2Lab))
    a_target, b_target = c.quantize_kmeans(a_target,b_target)      # quantized true-color image
    print "a, b sizes :"
    print np.shape(l)
    print np.shape(a)
    print np.shape(b)
    print np.shape(l_target)
    print np.shape(a_target)
    print np.shape(b_target)
    targetColorMap = cv2.merge((128*np.uint8(np.ones(np.shape(l_target))),np.uint8(a_target),np.uint8(b_target)))
    targetQuant = cv2.merge((np.uint8(l_target),np.uint8(a_target),np.uint8(b_target)))   

    a_err = pow(a - a_target,2)
    b_err = pow(b - b_target,2)

    errMap = a_err + b_err                          # error heatmap
    avgError = (np.sqrt(np.sum(a_err)) + np.sqrt(np.sum(b_err)))/(a_err.shape[0] * a_err.shape[1])  # total error metric

    print ('Average Error (2-norm) = %f'%avgError)
    print ('Maximum Error = %f'%max(errMap.flatten()))

    #now, display the original image, the BW image, and our colorized version
    fig = plt.figure(1)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(cv2.imread(training_files[0]), cv.CV_BGR2RGB))
    ax1.set_axis_off()
    ax1.set_title('Training Image')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(grayscale_image, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Grayscale')

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(colorized_image)
    ax3.set_axis_off()
    ax3.set_title('Colorized')


    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(cv2.cvtColor(newColorMap,cv.CV_Lab2RGB))
    ax4.set_axis_off()
    ax4.set_title('New colormap')

    # plt.savefig('/shared/users/asousa/ImageColorization/output_figure4.png')


    ax5 = fig.add_subplot(2,3,5)
    ax5.imshow(g, cmap='gray')
    ax5.set_axis_off()
    ax5.set_title('Color Variation')

    ax6 = fig.add_subplot(2,3,6)
    ax6.imshow(errMap, cmap='gray')
    ax6.set_axis_off()
    ax6.set_title('Prediction Error')


#    plt.savefig('ImageColorization/output_figure4.png')
    plt.show()

   
