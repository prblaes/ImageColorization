import cv, cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

from colorizer import Colorizer

def get_grayscale_from_color(color_file):
    '''
    Takes the path to a RGB image file and returns a numpy array of its luminance
    '''
    L, _, _ = cv2.split(cv2.cvtColor(cv2.imread(color_file), cv.CV_BGR2Lab))
    return L

if __name__ == '__main__':
    training_files = ['images/houses/calhouse_0001.jpg' ]
    input_file = 'images/houses/calhouse_0007.jpg'

    c = Colorizer(probability=False)
    img = get_grayscale_from_color(input_file)

    #load saved colorization data (pre-graphcut)
    f = open('dump.dat', 'rb')
    d = pickle.load(f)
    f.close()

    #inject data into colorizer object
    c.colors_present = d['colors']
    c.g = d['g']
    c.label_to_color_map = d['cmap']

    for l in range(0, 15, 3):
        print('l=%d'%l)
        output_labels = c.graphcut(d['S'], l=l)
        
        print np.mean(output_labels)
        #convert labels to rgb colors
        m,n = c.g.shape
        output_a = np.zeros((m,n))
        output_b = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                a,b = d['cmap'][d['colors'][output_labels[i,j]]]
                output_a[i,j] = a
                output_b[i,j] = b

        output_img = cv2.cvtColor(cv2.merge((img, np.uint8(output_a), np.uint8(output_b))), cv.CV_Lab2RGB)


        #save the outputs
        cv2.imwrite('output/out%d.jpg'%l, cv2.cvtColor(output_img, cv.CV_RGB2BGR))
    cv2.imwrite('output/g.jpg', np.uint8(255*d['g']/np.max(d['g'])))

