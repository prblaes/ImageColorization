#!/usr/bin/env python

import cv, cv2
import matplotlib.pyplot as plt

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

    #training_files = ['images/houses/calhouse_0001.jpg' ]
    #input_file = 'images/houses/calhouse_0002.jpg'

    training_files = ['images/book_chapter/islande.jpg' ]
    input_file = 'images/book_chapter/paysage_gris.png'

    #training_files = ['images/cats/cat.jpg' ]
    #input_file = 'images/cats/cats3.jpg'
    
    c = Colorizer()

    #train the classifiers
    c.train(training_files)

    #for now, convert an already RGB image to grayscale for our input
    grayscale_image = get_grayscale_from_color(input_file)

    #colorize the input image
    colorized_image = c.colorize(grayscale_image)

    #save the outputs
    cv2.imwrite('output_gray.jpg', grayscale_image)
    cv2.imwrite('output_color.jpg', cv2.cvtColor(colorized_image, cv.CV_RGB2BGR))


    #now, display the original image, the BW image, and our colorized version
    fig = plt.figure(1)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(cv2.cvtColor(cv2.imread(training_files[0]), cv.CV_BGR2RGB))
    ax1.set_axis_off()
    ax1.set_title('Training Image')


    #ax2 = fig.add_subplot(1, 4, 2)
    #ax2.imshow(cv2.cvtColor(cv2.imread(input_file), cv.CV_BGR2RGB))
    #ax2.set_axis_off()
    #ax2.set_title('Original RGB')

    ax3 = fig.add_subplot(1, 3, 2)
    ax3.imshow(grayscale_image, cmap='gray')
    ax3.set_axis_off()
    ax3.set_title('Grayscale')

    ax4 = fig.add_subplot(1, 3, 3)
    ax4.imshow(colorized_image)
    ax4.set_axis_off()
    ax4.set_title('Colorized')

    plt.show()


   
