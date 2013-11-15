
import numpy as np
import cv
import cv2
import itertools
import sklearn

SURF_WINDOW = 20
windowSize = 10

class Colorizer(object):
    '''
    TODO: write docstring...
    '''

    def __init__(self):
        self.levels = 256
        self.colors = np.arange(self.levels**2)
        # declare classifiers
        self.SVMs = []
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
        pass


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
        

    def colorize(self, grayscaleImage):
        '''
        -- colorizes a grayscale image, using the set of SVMs defined by train().

        Returns:
        -- ndarray(m,n,3): a mxn pixel RGB image
        '''
    
        #pass through the grayscale image for now as a RGB image...
        return cv2.merge((grayscaleImage, grayscaleImage, grayscaleImage))


    def load_image(self, path):
        '''
        Read in a file and separate into L*a*b* channels
        '''
        
        #read in original image
        self.img = cv2.cvtColor(cv2.imread(path), cv.CV_BGR2RGB)

        #convert to L*a*b* space and split into channels
        self.l, self.a, self.b = cv2.split(cv2.cvtColor(self.img, cv.CV_RGB2Lab))

    def posterize(self, num_levels):
        '''
        Quantizes each channel of the currently loaded image into `num_levels' distinct levels.
        
        credit: http://stackoverflow.com/a/11072667
        '''
        inds = np.arange(0, 256)
        div = np.linspace(0, 255, num_levels+1)[1]
        quantiz = np.int0(np.linspace(0, 255, num_levels))
        color_levels = np.clip(np.int0(inds/div), 0, num_levels-1)
        palette = quantiz[color_levels]
        #self.l = cv2.convertScaleAbs(palette[self.l])
        self.a = cv2.convertScaleAbs(palette[self.a])
        self.b = cv2.convertScaleAbs(palette[self.b])
        c.levels = num_levels

        a_unique = np.unique(self.a)
        b_unique = np.unique(self.b)

        self.colors = list(itertools.product(a_unique, b_unique))


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


