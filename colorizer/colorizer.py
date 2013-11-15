from __future__ import division
import numpy as np
import cv
import cv2
import itertools
from sklearn.svm import SVC
import sys
import pdb

SURF_WINDOW = 20
windowSize = 10
NTRAIN = 15000 #number of random pixels to train on

class Colorizer(object):
    '''
    TODO: write docstring...
    '''

    def __init__(self, ncolors=64):
       
        #number of bins in the discretized a,b channels
        self.levels = int(np.floor(np.sqrt(ncolors)))
        self.ncolors = self.levels**2 #recalculate ncolors in case the provided parameter is not a perfect square
        
        #generate color palette for discrities Lab space
        self.discretize_color_space()

        # declare classifiers
        self.svm = [SVC() for i in range(self.ncolors)]
        self.colors_present = np.zeros(len(self.colors))
        self.surf = cv2.DescriptorExtractor_create('SURF')
        self.surf.setBool('extended', True) #use the 128-length descriptors

    def feature_surf(self, img, pos):
        '''
        Gets the SURF descriptor of img at pos = (x,y).
        Assume img is a single channel image.
        '''
        kp = cv2.KeyPoint(pos[0], pos[1], SURF_WINDOW)
        _, des = self.surf.compute(img, [kp])
        return des[0]

    def train(self, files):
        '''
        -- Reads in a set of training images. 
        -- Converts from RGB to LAB colorspace.
        -- Extracts feature vectors at each pixel of each training image.
        -- (complexity reduction?).
        -- Train a set of SVMs on the dataset (one vs. others classifiers, per each of nColors output colors)
        -- writes to class array of SVM objects.
        '''

        features = []
        classes = []

        for f in files:

            l,a,b = self.load_image(f)

            #quantize the a, b components
            a,b = self.posterize(a,b)
            cv2.imshow('a',a)
            cv2.imshow('b',b)
            #dimensions of image
            m,n = l.shape 

            #extract features from training image
            for i in xrange(NTRAIN):

                #choose random pixel in training image
                x = int(np.random.uniform(n))
                y = int(np.random.uniform(m))
            
                meanvar = np.array([self.getMean(l, (x,y))])#, self.getVariance(l, (x,y))]) #variance is giving NaN

                feat = self.feature_surf(l, (x,y)) #np.concatenate((meanvar, self.feature_surf(l, (x,y))))

                features.append(feat)
                classes.append(self.color_to_label_map[(a[y,x], b[y,x])])

        features = np.array(features)
        classes = np.array(classes)
    
        #train the classifiers
        try: 
            for i in xrange(self.ncolors):
                sys.stdout.write('\rtraining svm #%d'%i)
                sys.stdout.flush()
                y = np.array([1 if j==True else -1 for j in classes==i]) #generate +/-1 labels for this classifier
               
                #if the i^th color is actually present in the training image, train the corresponding classifier
                if -1*len(y) != np.sum(y):
                    self.svm[i].fit(features, y)
                    self.colors_present[i] = 1

        except Exception, e:
            pdb.set_trace()
            
        print('')
        
        #pdb.set_trace()

    def getMean(self, img, loc):
        ''' 
        Returns mean value over a windowed region around (x,y)
        '''

        xlim = (max(loc[0] - windowSize,0), min(loc[0] + windowSize,img.shape[0]))
        ylim = (max(loc[1] - windowSize,0), min(loc[1] + windowSize,img.shape[1]))

        return np.mean(img[xlim[0]:xlim[1],ylim[0]:ylim[1]])

        

    def getVariance(self, img, loc):

        xlim = (max(loc[0] - windowSize,0), min(loc[0] + windowSize,img.shape[0]))
        ylim = (max(loc[1] - windowSize,0), min(loc[1] + windowSize,img.shape[1]))

        return np.var(img[xlim[0]:xlim[1],ylim[0]:ylim[1]])
        

    def colorize(self, img):
        '''
        -- colorizes a grayscale image, using the set of SVMs defined by train().

        Returns:
        -- ndarray(m,n,3): a mxn pixel RGB image
        '''

        m,n = img.shape

        num_classified = 0
        
        output_a = np.zeros(img.shape)
        output_b = np.zeros(img.shape)
        count=0
        for x in xrange(int(n/4), int(n/2)):
            for y in xrange(int(m/4), int(m/2)):
                feat = np.array([self.feature_surf(img, (x,y)) ])
                sys.stdout.write('\r%3.3f%%'%(100*count/(m*n)))
                sys.stdout.flush()
                count += 1
                for i in xrange(self.ncolors):
                    if self.colors_present[i]:
                        if self.svm[i].predict(feat)==1:
                            a,b = self.label_to_color_map[i]
                            output_a[y,x] = a
                            output_b[y,x] = b
                            num_classified += 1
        
        output_img = cv2.cvtColor(cv2.merge((img, np.uint8(output_a), np.uint8(output_b))), cv.CV_Lab2RGB)

        print('\nclassified %d\n'%num_classified)

        return output_img


    def load_image(self, path):
        '''
        Read in a file and separate into L*a*b* channels
        '''
        
        #read in original image
        img = cv2.imread(path)

        #convert to L*a*b* space and split into channels
        l, a, b = cv2.split(cv2.cvtColor(img, cv.CV_BGR2Lab))

        return l, a, b


    def discretize_color_space(self):
        '''
        Generates self.palette, which maps the 8-bit color channels to bins in the discretized space.
        Also generates self.colors, which is a list of all possible colors (a,b components) in this space.
        '''
        inds = np.arange(0, 256)
        div = np.linspace(0, 255, self.levels+1)[1]
        quantiz = np.int0(np.linspace(0, 255, self.levels))
        color_levels = np.clip(np.int0(inds/div), 0, self.levels-1)
        self.palette = quantiz[color_levels]
        bins = np.unique(self.palette) #the actual color bins
        self.colors = list(itertools.product(bins, bins))
        self.color_to_label_map = {c:i for i,c in enumerate(self.colors)} #this maps the color pair to the index of the color
        self.label_to_color_map = dict(zip(self.color_to_label_map.values(),self.color_to_label_map.keys()))


    def posterize(self, a, b):
        a_quant = cv2.convertScaleAbs(self.palette[a])
        b_quant = cv2.convertScaleAbs(self.palette[b])
        return a_quant, b_quant



#Make plots to test image loading, Lab conversion, and color channel quantization.
#Should probably write as a proper unit test and move to a separate file.
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    c = Colorizer()
    c.load_image('../test/cat.jpg')
    
    #plot the original
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(c.img)
    ax.set_axis_off()
    ax.set_title('Original RGB Image')
   
    #helper for plotting the Lab image components
    def plot_lab(idx):
        fig = plt.figure(idx, figsize=(10, 20))
        ax = fig.add_subplot(4,1,1)
        ax.imshow(cv2.merge((c.l, c.a, c.b)))
        ax.set_axis_off()
        ax.set_title('L*a*b* Image, %d colors'%(len(c.colors)))

        ax = fig.add_subplot(4,1,2)
        ax.imshow(c.l, cmap='gray')
        ax.set_axis_off()
        ax.set_title('L* Channel')

        ax = fig.add_subplot(4,1,3)
        ax.imshow(c.a, cmap='gray')
        ax.set_axis_off()
        ax.set_title('a* Channel')    
        
        ax = fig.add_subplot(4,1,4)
        ax.imshow(c.b, cmap='gray')
        ax.set_axis_off()
        ax.set_title('b* Channel')
        plt.subplots_adjust(hspace=0.3)

    #plot unquantized Lab components
    #plot_lab(2)

    #2 levels
    #c.posterize(2)
    #plot_lab(3)

    #8 levels
    #c.load_image('../test/cat.jpg') #reload because last posterize call overwrote originals
    #c.posterize(8)
    #plot_lab(4)

    #16 levels
    c.load_image('../test/cat.jpg')
    c.posterize(16)
    #plot_lab(5)

    #plt.show()
   
    blank = np.zeros((100,100))
    print blank

    print type(blank)
    print type(c.img)
    ax.imshow(blank)
    plt.show()
    print c.getMean(c.l,(150,220))
    print c.getVariance(blank,(0,0))


