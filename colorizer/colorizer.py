
from __future__ import division
import numpy as np
import cv
import cv2
import itertools
from sklearn.svm import SVC
import sys
import pdb
from scipy.fftpack import dct

SURF_WINDOW = 20
DCT_WINDOW = 20
windowSize = 10
gridSpacing = 2

NTRAIN = 5000 #number of random pixels to train on

class Colorizer(object):
    '''
    TODO: write docstring...
    '''

    def __init__(self, ncolors=128, random=False, probability=False):
       
        #number of bins in the discretized a,b channels
        self.levels = int(np.floor(np.sqrt(ncolors)))
        self.ncolors = self.levels**2 #recalculate ncolors in case the provided parameter is not a perfect square
        
        #generate color palette for discrities Lab space
        self.discretize_color_space()

        # declare classifiers
        self.svm = SVC(probability=probability, gamma=0.1)

        self.probability = probability
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

    def feature_dct(self, img, pos):
        pass

    def feature_position(self, img, pos):
        m,n = img.shape
        x_pos = pos[0]/n
        y_pos = pos[1]/m

        return np.array([x_pos, y_pos])

    def feature_histogram(self, img, pos, nbins=10):
        xlim = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,img.shape[1]))
        ylim = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,img.shape[0]))
        patch = img[ylim[0]:ylim[1],xlim[0]:xlim[1]]
         
        return patch       
            

    def get_features(self, img, pos):
        intensity = np.array([img[pos[1], pos[0]]])
        position = self.feature_position(img, pos)
        meanvar = np.array([self.getMean(img, pos), self.getVariance(img, pos)]) #variance is giving NaN
        feat = np.concatenate((position, meanvar, self.feature_surf(img, pos)))

        return feat



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
        numTrainingExamples = 0
        for f in files:

            l,a,b = self.load_image(f)

            #quantize the a, b components
            a,b = self.posterize(a,b)

            #dimensions of image
            m,n = l.shape 

            #extract features from training image
            # (Select uniformly-spaced training pixels)
            for x in xrange(int(gridSpacing/2),n,gridSpacing):
                for y in xrange(int(gridSpacing/2),m,gridSpacing):
                    sys.stdout.write('\rgenerating feature: %3.0f'%(numTrainingExamples))
                    sys.stdout.flush()
                    
                    features.append(self.get_features(l, (x,y)))
                    classes.append(self.color_to_label_map[(a[y,x], b[y,x])])

                    numTrainingExamples = numTrainingExamples + 1

        features = np.array(features)
        classes = np.array(classes)

        #train the classifiers
        print " "
        print "Training SVM" 
        try: 
            self.svm.fit(features,classes)

        except Exception, e:
            pdb.set_trace()
            
        print('')
        

    def getMean(self, img, pos):
        ''' 
        Returns mean value over a windowed region around (x,y)
        '''

        xlim = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,img.shape[1]))
        ylim = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,img.shape[0]))

        return np.mean(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])

        

    def getVariance(self, img, pos):

        xlim = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,img.shape[1]))
        ylim = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,img.shape[0]))

        return np.var(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])/10000
        

    def colorize(self, img, skip=1):
        '''
        -- colorizes a grayscale image, using the set of SVMs defined by train().

        Returns:
        -- ndarray(m,n,3): a mxn pixel RGB image
        '''

        m,n = img.shape

        num_classified = 0

        _,output_a,output_b = cv2.split(cv2.cvtColor(cv2.merge((img, img, img)), cv.CV_RGB2Lab)) #default a and b for a grayscale image
        
        count=0
        for x in xrange(0,n,skip):
            for y in xrange(0,m,skip):

                feat = self.get_features(img, (x,y))

                sys.stdout.write('\rcolorizing: %3.3f%%'%(np.min([100, 100*count*skip**2/(m*n)])))
                sys.stdout.flush()
                count += 1

                if self.probability:
                    probs = [self.svm[i].predict_proba(feat) for i in xrange(self.ncolors)] #calc the probability for each color in cspace
                    ml_color = np.argmax(probs)[0] #choose the best color
                    a,b = self.label_to_color_map[ml_color]

                    output_a[y-int(skip/2):y+int(skip/2)+1,x-int(skip/2):x+int(skip/2)+1] = a
                    output_b[y-int(skip/2):y+int(skip/2)+1,x-int(skip/2):x+int(skip/2)+1] = b
                    num_classified += 1

                else:
                    a,b = self.label_to_color_map[int(self.svm.predict(feat)[[0]])]

                    output_a[y-int(skip/2):y+int(skip/2)+1,x-int(skip/2):x+int(skip/2)+1] = a
                    output_b[y-int(skip/2):y+int(skip/2)+1,x-int(skip/2):x+int(skip/2)+1] = b

                    num_classified += 1

        output_img = cv2.cvtColor(cv2.merge((img, np.uint8(output_a), np.uint8(output_b))), cv.CV_Lab2RGB)
    
        print('\nclassified %d%%\n'% (100*num_classified*(skip**2)/(m*n)))

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
        self.colors = list(itertools.product(bins, bins)) #find all permutations of a/b bins
        self.color_to_label_map = {c:i for i,c in enumerate(self.colors)} #this maps the color pair to the index of the color
        self.label_to_color_map = dict(zip(self.color_to_label_map.values(),self.color_to_label_map.keys())) #takes a label and returns a,b


    def posterize(self, a, b):
        a_quant = cv2.convertScaleAbs(self.palette[a])
        b_quant = cv2.convertScaleAbs(self.palette[b])
        return a_quant, b_quant

    
    def get_edges(self, img, blur_width=2):
        img_blurred = cv2.GaussianBlur(img, (0, 0), blur_width)
        vh = cv2.Sobel(img_blurred, -1, 1, 0)
        vv = cv2.Sobel(img_blurred, -1, 0, 1)
        return vv, vh



#Make plots to test image loading, Lab conversion, and color channel quantization.
#Should probably write as a proper unit test and move to a separate file.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    training_files = ['images/houses/calhouse_0007.jpg' ]

    c = Colorizer()

    c.train(training_files)


