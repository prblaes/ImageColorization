
import numpy as np
import cv
import cv2
import itertools

class Colorizer(object):
    '''
    TODO: write docstring...
    '''

    def __init__(self):
        self.levels = 256
        self.colors = np.arange(self.levels**2)

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
    ax.imshow(c.img)
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
    plot_lab(2)

    #2 levels
    c.posterize(2)
    plot_lab(3)

    #8 levels
    c.load_image('../test/cat.jpg') #reload because last posterize call overwrote originals
    c.posterize(8)
    plot_lab(4)

    #16 levels
    c.load_image('../test/cat.jpg')
    c.posterize(16)
    plot_lab(5)

    plt.show()
   

