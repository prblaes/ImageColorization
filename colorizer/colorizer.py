
from __future__ import division
import numpy as np
import cv
import cv2
import itertools
from sklearn.svm import SVC
from sklearn import preprocessing
import sys
import pdb
from scipy.fftpack import dct
from gco_python import pygco
from scipy.cluster.vq import kmeans,vq
from sklearn.decomposition import PCA
import scipy.ndimage.filters
import pickle

SURF_WINDOW = 20
DCT_WINDOW = 20
windowSize = 10
gridSpacing = 7

SAVE_OUTPUTS = True

NTRAIN = 1000 #number of random pixels to train on

NPCA = 30 # size of the reduced 

class Colorizer(object):
    '''
    TODO: write docstring...
    '''

    def __init__(self, ncolors=16, random=False, probability=False):
       
        #number of bins in the discretized a,b channels
        self.levels = int(np.floor(np.sqrt(ncolors)))
        #self.ncolors = self.levels**2 #recalculate ncolors in case the provided parameter is not a perfect square
        self.ncolors = ncolors
        
        #generate color palette for discrities Lab space
        #self.discretize_color_space()

        # declare classifiers
        #self.svm = SVC(probability=probability, gamma=0.1)
        self.svm = [SVC(probability=probability, gamma=0.1) for i in range(self.ncolors)]

        self.scaler = preprocessing.MinMaxScaler()                          # Scaling object -- Normalizes feature array
        
        self.pca = PCA(NPCA)

        self.probability = probability
        self.colors_present = []
        self.surf = cv2.DescriptorExtractor_create('SURF')
        self.surf.setBool('extended', True) #use the 128-length descriptors

    def feature_surf(self, img, pos):
        '''
        Gets the SURF descriptor of img at pos = (x,y).
        Assume img is a single channel image.
        '''
        octave2 = cv2.GaussianBlur(img, (0, 0), 1)
        octave3 = cv2.GaussianBlur(img, (0, 0), 2)
        kp = cv2.KeyPoint(pos[0], pos[1], SURF_WINDOW)
        _, des1 = self.surf.compute(img, [kp])
        _, des2 = self.surf.compute(octave2, [kp])
        _, des3 = self.surf.compute(octave3, [kp])

        return np.concatenate((des1[0], des2[0], des3[0]))

    def feature_dft(self, img, pos):
        x = 1
   
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
        #laplacian = self.getLaplacian(img,pos)
#        feat = np.concatenate((position, meanvar, self.feature_surf(img, pos)))
        #feat = np.concatenate((meanvar, self.feature_surf(img, pos)))
        feat = np.concatenate((position, meanvar, self.feature_surf(img, pos)))
        #print feat
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
        self.local_grads = []
        classes = []
        numTrainingExamples = 0
        for f in files:

            l,a,b = self.load_image(f)

            self.compute_gradients(a,b)

            a,b = self.posterize_kmeans(a,b,self.ncolors)

            #quantize the a, b components
            #a,b = self.posterize(a,b)

            #dimensions of image
            m,n = l.shape 


            #extract features from training image
            # (Select uniformly-spaced training pixels)
            #for x in xrange(int(gridSpacing/2),n,gridSpacing):
            #    for y in xrange(int(gridSpacing/2),m,gridSpacing):
            #extract features from training image
            for i in xrange(NTRAIN):
                #choose random pixel in training image
                x = int(np.random.uniform(n))
                y = int(np.random.uniform(m))

                sys.stdout.write('\rgenerating feature: %3.0f'%(numTrainingExamples))
                sys.stdout.flush()
                    
                features.append(self.get_features(l, (x,y)))
                classes.append(self.color_to_label_map[(a[y,x], b[y,x])])
                    
                #save vertical/horizontal color gradients at training feature locations
                self.local_grads.append(self.grad[y,x])

                numTrainingExamples = numTrainingExamples + 1

        # normalize columns
        self.features = self.scaler.fit_transform(np.array(features))
        classes = np.array(classes)

        # reduce dimensionality
        self.features = self.pca.fit_transform(self.features)

        #train the classifiers
        print " "
        print "Training SVM" 
        try: 
            for i in range(self.ncolors):
                if len(np.where(classes==i)[0])>0:
                    curr_class = (classes==i).astype(np.int32)
                    self.colors_present.append(i)
                    self.svm[i].fit(self.features,(classes==i).astype(np.int32))

        except Exception, e:
            pdb.set_trace()

            
        print " "
        # print the number of support vectors for each class
        #print "Number of support vectors: ", self.svm.n_support_
        #pdb.set_trace()
        print('')

    def compute_gradients(self, a, b):
        grad_a_horiz = cv2.Sobel(a, -1, 1, 0)
        grad_a_vert = cv2.Sobel(a, -1, 0, 1)

        grad_a = 0.5*grad_a_horiz + 0.5*grad_a_vert
        
        grad_b_horiz = cv2.Sobel(b, -1, 1, 0)
        grad_b_vert = cv2.Sobel(b, -1, 0, 1)

        grad_b = 0.5*grad_b_horiz + 0.5*grad_b_vert

        #self.gradv = np.sqrt(grad_a_vert**2 + grad_b_vert**2) #vertical grad magnitude
        #self.gradh = np.sqrt(grad_a_horiz**2 + grad_b_horiz**2) #horizontal grad magnitude

        self.grad = np.sqrt(grad_a**2 + grad_b**2)


    def color_variation(self, feat, sigma=2):
        m,n = self.features.shape
        
        #exponential kernel function
        def k(w,v):
            return np.exp(-1*np.linalg.norm(w - v)**2 / (2*sigma**2))
        
        g_num=0

        g_den=0

        #only use every 4 features for this KDE. Very slow otherwise...
        for i in range(1, m, 4):
            x = k(self.features[i,:], feat)
            g_num += x * self.local_grads[i]
            g_den += x

        g = g_num / g_den

        return g
        
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

        return np.var(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])/1000 #switched to Standard Deviation --A
        
    def getLaplacian(self, img, pos):

        xlim = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,img.shape[1]))
        ylim = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,img.shape[0]))

        lap = scipy.ndimage.filters.laplace(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])
        return np.ravel(lap)

        

    def colorize(self, img, skip=4):
        '''
        -- colorizes a grayscale image, using the set of SVMs defined by train().

        Returns:
        -- ndarray(m,n,3): a mxn pixel RGB image
        '''

        m,n = img.shape

        num_classified = 0

        _,raw_output_a,raw_output_b = cv2.split(cv2.cvtColor(cv2.merge((img, img, img)), cv.CV_RGB2Lab)) #default a and b for a grayscale image

        output_a = np.zeros(raw_output_a.shape)
        output_b = np.zeros(raw_output_b.shape)

        num_classes = len(self.colors_present)
        label_costs = np.zeros((m,n,num_classes))

        #self.g = np.zeros(raw_output_a.shape)
        
        count=0
        for x in xrange(0,n,skip):
            for y in xrange(0,m,skip):

                feat = self.scaler.transform(self.get_features(img, (x,y)))
                feat = self.pca.transform(feat)

                sys.stdout.write('\rcolorizing: %3.3f%%'%(np.min([100, 100*count*skip**2/(m*n)])))
                sys.stdout.flush()
                count += 1
                
                #self.g[y-int(skip/2):y+int(skip/2)+1,x-int(skip/2):x+int(skip/2)+1] = self.color_variation(feat)

                #get margins to estimate confidence for each class
                for i in range(num_classes):
                    cost = -1*self.svm[self.colors_present[i]].decision_function(feat)[0]
                    label_costs[y-int(skip/2):y+int(skip/2)+1,x-int(skip/2):x+int(skip/2)+1,i] = cost

        edges = self.get_edges(img)
        self.g = np.sqrt(edges[0]**2 + edges[1]**2)
        self.g = np.log10(self.g)
      
        if SAVE_OUTPUTS:
            #dump to pickle
            print('saving to dump.dat')
            fid = open('dump.dat', 'wb') 
            pickle.dump({'S': label_costs, 'g': self.g, 'cmap': self.label_to_color_map, 'colors': self.colors_present}, fid)
            fid.close()

        #postprocess using graphcut optimization 
        output_labels = self.graphcut(label_costs)
        
        for i in range(m):
            for j in range(n):
                a,b = self.label_to_color_map[self.colors_present[output_labels[i,j]]]
                output_a[i,j] = a
                output_b[i,j] = b

        output_img = cv2.cvtColor(cv2.merge((img, np.uint8(output_a), np.uint8(output_b))), cv.CV_Lab2RGB)
        print('\nclassified %d%%\n'% np.max([100,(100*num_classified*(skip**2)/(m*n))]))

        return output_img, self.g


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

    
    def get_edges(self, img, blur_width=3):
        img_blurred = cv2.GaussianBlur(img, (0, 0), blur_width)
        vh = cv2.Sobel(img_blurred, -1, 1, 0)
        vv = cv2.Sobel(img_blurred, -1, 0, 1)

        vh = vh/np.max(vh)
        vv = vv/np.max(vv)

        return vv, vh

    def graphcut(self, label_costs, l=100):

        num_classes = len(self.colors_present)
        
        #calculate pariwise potiential costs (distance between color classes)
        pairwise_costs = np.zeros((num_classes, num_classes))
        for ii in range(num_classes):
            for jj in range(num_classes):
                c1 = np.array(self.label_to_color_map[ii])
                c2 = np.array(self.label_to_color_map[jj])
                pairwise_costs[ii,jj] = np.linalg.norm(c1-c2)
        
        label_costs_int32 = (100*label_costs).astype('int32')
        pairwise_costs_int32 = (l*pairwise_costs).astype('int32')
        vv_int32 = (1/self.g).astype('int32')
        vh_int32 = (1/self.g).astype('int32')
        
        #perform graphcut optimization
        new_labels = pygco.cut_simple_vh(label_costs_int32, pairwise_costs_int32, vv_int32, vh_int32, n_iter=10, algorithm='swap') 

        #new_labels = pygco.cut_simple(label_costs_int32, pairwise_costs_int32, algorithm='swap')

        return new_labels
        

    def posterize_kmeans(self, a, b, k):
        w,h = np.shape(a)
        
        # reshape matrix
        pixel = np.reshape((cv2.merge((a,b))),(w * h,2))

        # cluster
        centroids,_ = kmeans(pixel,k) # six colors will be found
 
        # quantization
        qnt,_ = vq(pixel,centroids)

        # reshape the result of the quantization
        centers_idx = np.reshape(qnt,(w,h))
        clustered = centroids[centers_idx]

        #color-mapping lookup tables
        self.color_to_label_map = {c:i for i,c in enumerate([tuple(i) for i in centroids])} #this maps the color pair to the index of the color
        self.label_to_color_map = dict(zip(self.color_to_label_map.values(),self.color_to_label_map.keys())) #takes a label and returns a,b

        a_quant = clustered[:,:,0]
        b_quant = clustered[:,:,1]
        return a_quant, b_quant

    def posterize_external_image(self, a,b):
        w,h = np.shape(a)
        
        # reshape matrix
        pixel = np.reshape((cv2.merge((a,b))),(w * h,2))

        # quantization
        qnt,_ = vq(pixel,centroids)

        # reshape the result of the quantization
        centers_idx = np.reshape(qnt,(w,h))
        clustered = centroids[centers_idx]

        a_quant = clustered[:,:,0]
        b_quant = clustered[:,:,1]
        return a_quant, b_quant
    
#Make plots to test image loading, Lab conversion, and color channel quantization.
#Should probably write as a proper unit test and move to a separate file.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    training_files = ['images/houses/calhouse_0007.jpg' ]

    c = Colorizer()

    c.train(training_files)


