# File: joy.py 
# Purpose: Developed and collected for "Python Machine Lenaring"
# Author: Youngsup Kim, idebtor@gmail.com
# 2018.03.23 - creation
# Use imp.reload(joy) after modified and %%writefile

import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd
import sys
import os

import sklearn
import sklearn.datasets
import sklearn.linear_model
import h5py

def iris_data(standardized=False, shuffled=False): 
    try: 
        datafile = 'data/iris.data'   
        if os.path.isfile(datafile):
            df = pd.read_csv(datafile)
        else :
            datafile = 'https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data'
            df = pd.read_csv(datafile, header=None)
    except:
        sys.exit('Having a problem in reading {}'.format(filename))
    
    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 1, -1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values
    X = X[0:100]  # class 0 and class 1
    y = y[0:100]  # class 0 and class 1
    
    if standardized == True: 
        mu, sigma = X.mean(axis=0), X.std(axis=0)
        X = (X - mu) / sigma

    if shuffled == True:
        random_gen = np.random.RandomState(1)
        shuffle_idx = random_gen.permutation(len(y))
        X, y = X[shuffle_idx], y[shuffle_idx]
    
    return X, y

def joydata(standardized=False, shuffled=False):
    ''' joydata reads data/joydata.txt file and returns the data'
    Parameters
        standardized: If True, performs standardization or 
            feature scaling such that mean is 0 and std dev = 1.0
        shuffled: If True, performs random shuffling of samples
            Used a fixed random seed = 1 for reproducibility
    '''
    return getXy('data/joydata.txt', standardized=standardized, shuffled=shuffled)

def joy_data(standardized=False, shuffled=False):
    ''' joy_data reads data/joy_data.txt file and returns the data'
    Parameters
        standardized: If True, performs standardization or 
            feature scaling such that mean is 0 and std dev = 1.0
        shuffled: If True, performs random shuffling of samples
            Used a fixed random seed = 1 for reproducibility
    '''

    return getXy('data/joy_data.txt', standardized=standardized, shuffled=shuffled)

def joy_Ndata(standardized=False, shuffled=False):
    ''' joy_data reads data/joy_data.txt file and returns the data'
    Parameters
        standardized: If True, performs standardization or 
            feature scaling such that mean is 0 and std dev = 1.0
        shuffled: If True, performs random shuffling of samples
            Used a fixed random seed = 1 for reproducibility
    '''
    return getXy('data/joy_dataNoise.txt', standardized=standardized, shuffled=shuffled)

def toy_data(standardized=False, shuffled=False):
    ''' toy_data reads data/toy_data.txt file and returns the data'
    Parameters
        standardized: If True, performs standardization or 
            feature scaling such that mean is 0 and std dev = 1.0
        shuffled: If True, performs random shuffling of samples
            Used a fixed random seed = 1 for reproducibility
    '''
    return getXy('data/toy_data.txt', standardized=standardized, shuffled=shuffled)



def getXy(filename, bipolar=True, standardized=False, shuffled=False):
    ''' reads the 'filename' file which has a list of three columns data, 
       x1, x2, classlabel, and returns the data, X(x1, x2) and y
    Parameters
        bipolar: If True, returns that y consists of either 1 or -1
            otherwise, it returns as they are read
        standardized: If True, performs standardization or 
            feature scaling such that mean is 0 and std dev = 1.0
        shuffled: If True, performs random shuffling of samples
            Used a fixed random seed = 1 for reproducibility
    '''
    
    try:
        data = np.genfromtxt(filename)
    except:
        sys.exit('Having a problem in reading {}'.format(filename))
    
    x, y = data[:, :2], data[:, 2]
    y = y.astype(np.int)

    if shuffled:
        random_gen = np.random.RandomState(1)
        shuffle_idx = random_gen.permutation(len(y))
        x, y = x[shuffle_idx], y[shuffle_idx]
    
    if standardized: 
        mu, sigma = x.mean(axis=0), x.std(axis=0)
        x = (x - mu) / sigma
        
    if bipolar: 
        y[y != 1] = -1
        
    return x, y 


def noisy_circles(display = False, random_seed = 1):  
    N = 200
    X, Y = sklearn.datasets.make_circles(n_samples=N, factor=.5, 
                                         noise=.3, random_state = random_seed)
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    #print('X.shape={}, Y.shape={}'.format(X.shape, Y.shape))
    if display:
        plt.scatter(X[0, :], X[1, :], c=Y.squeeze(), s=40, cmap=plt.cm.Spectral);
    return X, Y
    
    
def noisy_moons(display = False, random_seed = 1):  
    N = 200
    X, Y = sklearn.datasets.make_moons(n_samples=N, 
                                       noise=.2, random_state=random_seed)
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    #print('X.shape={}, Y.shape={}'.format(X.shape, Y.shape))
    if display:
        plt.scatter(X[0, :], X[1, :], c=Y.squeeze(), s=40, cmap=plt.cm.Spectral);
    return X, Y


def blobs(display = False, random_seed = 1):  
    N = 200
    X, Y = sklearn.datasets.make_blobs(n_samples=N, random_state = random_seed, 
                                       n_features=2, centers=6)
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    Y = Y%2
    #print('X.shape={}, Y.shape={}'.format(X.shape, Y.shape))
    if display:
        plt.scatter(X[0, :], X[1, :], c=Y.squeeze(), s=40, cmap=plt.cm.Spectral);
    return X, Y



def planar_data(random_seed = 1):  
    """The data looks like a "flower" with some red (label y=0) and 
        some blue (y=1) points. 
        The original source: Andrew Ng's ML course in Coursera
    """
    np.random.seed(random_seed)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    return X.T, Y.T


def gaussian_quantiles(display = False, random_seed = 1):
    N = 200
    X, Y = sklearn.datasets.make_gaussian_quantiles(mean=None, 
                    cov=0.5, n_samples=N, n_features=2, n_classes=2, 
                    shuffle=True, random_state=random_seed)
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    Y = Y%2
    #print('X.shape={}, Y.shape={}'.format(X.shape, Y.shape))
    if display:
        plt.scatter(X[0, :], X[1, :], c=Y.squeeze(), s=40, cmap=plt.cm.Spectral);
    return X, Y





import struct
import csv
import os.path
import gzip
import pickle

try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')

mnist_url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

#dataset_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.abspath('./data')
save_file = dataset_dir + "/mnist.pkl"
fashion_save_file = dataset_dir + "/fashionmnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (28, 28)         # not (1, 28, 28)
img_size = 784

def _download(url_base, file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ", end='')
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
        _download(mnist_url_base, v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...", end='')
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...", end='')    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...", end='')
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

"""
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T
"""    

def load_mnist(normalize=True, flatten=True):
    """MNIST 데이터셋 읽기
    
    Parameters
    ------------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다. 
               True이면, (m, 784)형상으로 반환한다. 
               False이면, (m, 28, 28)형상으로 반환한다. 
    Returns
    ---------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    """        
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    """
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 28, 28)  

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 




import scipy.ndimage
def append_mnist_rotation(X, y, n_images, degree):
    """ X -- images, shape(m, 784), m is the number of images 
        y -- labels, shape(m, )
        n_images -- number of images randomly selected to rotate
        degree -- rotation angle
        
        Returns:
        X -- images, shape(m + n_images, 784), randomly selected 
              n_images rotated and appended to the end of X
        y -- labels, shape(m + n_images, ) 
              labels appended according to rotated images 
              
        Author: idebtor@gmail.com
        2018/05/01 - Created
    """
    X = X.reshape(-1, 28, 28)
    selected = np.random.choice(X.shape[0], n_images)
    Xr = X[selected]
    yr = y[selected]
    Xr = scipy.ndimage.rotate(Xr, 15.0, axes=(2, 1), cval=0.01, order=1, reshape=False)
    X = np.concatenate((X, Xr), axis = 0)
    y = np.concatenate((y, yr), axis = 0)
    return X.reshape(-1, 784), y



def _read_mnist(dataset = "training"):
    """
    A helper function to be used internally.
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples the first element being a numpy.uint8 2D array of pixel data for 
    the given image and with the second element being the label.
    
    This script uses a generator such that data will be read later upon requests.
    It runs fast at the beginning and use less memory. 
    
    Author: idebtor@gmail.com 
    2018/05/31 - Created
    """

    path = "./data"
    
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img_lbl = lambda idx: (img[idx], lbl[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img_lbl(i)
        




def save_mnist_csv(X, y, filename):
    """
    Parameters: 
    filename: 자료를 저장할 파일이름 (폴더 이름 data 없이)(자동으로 data/ 폴더 이름 추가됨)
    
    Results:  
      data/filename 으로 저장됨(형식은 csv, 첫 바이트는 레이블, 그 이후는 이미지 자료)
    """
    
    if filename is None:
        print("save_mnist_csv: A filename should be given including .csv extension.")
    
    if display == True:
        norm = False
    else:
        norm = True

    n_items = len(y)
          
    fullfilename = os.path.join('data', filename)
    print('Writing images({})...'.format(fullfilename), end='', flush=True)

    with open(fullfilename, 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\r')
        for item in range(n_items):
            if item % 100 == 0: print('.', end='', flush=True)
            mydata = X[item][:]              ### copying list - mydata.shape (1, 28, 28)
            mydata = mydata.flatten()         

            mydata = np.concatenate([[y[item]], mydata])  ### insert label in front ####
            mydata.astype(int)
            writer.writerow(mydata)
        
    #print('X len={}, type={}, shape={}, y len={}, type={}, shape={}, mydata={}'.
           #format(len(X), type(X), X.shape, len(y), type(y), y.shape, type(mydata)))   
    pass    
        
         
                

def read_mnist_csv(filename, display = True):
    """
    Reads the MNSIT csv file and returns X and y
    MNIST csv file may be written by using joy.save_mnist_csv() 
    
    filename -- csv filename without 'data' folder name
    display -- if False, X is normalized and (m, 28, 28)
    
    Returns:
    X -- images, shape(m, 28, 28) by default, display = True
          images, shape(m, 784)
          if display is False, X is normalized between 0 and 1
    y -- labels, shape(m, )
    
    Author: idebtor@gmail.com 
    2018/05/31 - Created
    """
    
    filename = os.path.join('data', filename)
    print('Reading images({})...'.format(filename), end='', flush=True)
    datafile = open(filename, 'r')
    datalist = datafile.readlines()           # ["5,0,.....", "0,0,.....", ...., "8,0,...."]
    datafile.close()
    print(len(datalist))
    
    intlist = []
    for i in range(len(datalist)):             #  [[5,0, .....], [0,0.....], ...., [8,0.....]]
        intlist.append([int(x) for x in datalist[i].split(',')])   
    n_images = len(datalist)
    
    y = np.zeros((n_images))
    X = np.zeros((n_images, 28, 28))
    
    for i in range(len(datalist)):
        y[i] = intlist[i][0]
        X[i] = np.array(intlist[i][1:]).reshape(28, 28)

    if not display:                          # shape (m, 784)
        X = X.reshape(n_images, -1)
        X= (X / 255.0 * 0.99) + 0.01           # normalized
    
    print('csv normalized X.shape={}, y.shape={}'.format(X.shape, y.shape))
    return X, y 




def load_mnist_num(n):
    """returns the first image data that has the label n in the MNIST"""
    (X, y), (_, __) = load_mnist()
    for i in range(len(y)):
        if y[i] == n:
            break
    return (X[i], y[i])



def load_mnist_weight(file_name):
    """returns the weight data that can be used for the prediction."""
    with open(file_name, 'rb') as f:
        w = pickle.load(f)
    return w
    
    
def save_mnist_weight(w, file_name):
    """saves the weight data for the MNIST model.
    w: weights to save
    file_name: to save
    """
    dir = 'data/' + file_name + '.weights'
    with open(dir, 'wb') as f:
        pickle.dump(w, f)


    

def show_mnist(image, inverted = False, savefig = None):
    """
    Render a given numpy.uint8 2D array of pixel data.
    image -- 2d image
    
    Author: idebtor@gmail.com 
    2018/05/31 - Created
    """
   
    if image.ndim == 3:
        print('Use show_mnist_grid() for multiple-image display')
        print('Expecting the shape (28, 28) for image, but got {}.'.format(image.shape))
        return

    #fig = pyplot.figure(figsize=(figsize, figsize))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if inverted:
        ax.imshow(image, cmap=plt.cm.Greys, interpolation=None)    
    else:
        ax.imshow(image, cmap=plt.cm.Greys_r, interpolation=None)
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', dpi=150)
    
    
def show_mnist_grid(images, inverted = False, figsize = 10, ncols=10, maxgrid = 100):
    """Render a list of images with given numpy.uint8 2D array of pixel data.
      
    images -- multiple of 2d-images in 3d format such as 
        (60000, 28, 28) shape case invokes showMNISTgrid()
    inverted -- if True, image is inverted, black to white, white to black 
    ncols -- number of columns in image display grid
    maxgrid -- total number of image to display in grid
    
    Author: idebtor@gmail.com 
    2018/05/31 - Created
    """
    #print('image len={}, type={}, shape={}'.
    #      format(len(images), type(images), images.shape))
    
    if images.ndim == 2:
        print('Use show_mnist() for a single-image display or check its shape.')
        print('Expecting the shape (m, 28, 28) for images, but got {}.'.format(images.shape))
        return
    
    fig = plt.figure(figsize=(figsize, figsize))
    n_images = len(images)
    if n_images > maxgrid: n_images = maxgrid
    
    row = n_images/ncols
    if (n_images % ncols != 0): row += 1
    for i in range(n_images):
        fig.add_subplot(row, ncols, i + 1)
        if inverted:
            plt.imshow(images[i], cmap=plt.cm.Greys)   
        else:
            plt.imshow(images[i], cmap=plt.cm.Greys_r)
        plt.axis('off')
    plt.tight_layout() 



def plot_xyc(x, y, clf=None, X0=False, annotate=False, xylabels=("$x_1$", "$x_2$"), 
             classes=['class1', 'class2'], savefig=None):
    
    """ plots training input data x and its class label y as well as the the linear 
        decision boundary and the value W[-1] or w in the first subplot if clf is given.
        In the second subplot, it displays the traning result. For the Perceptron, it displays
        epochs vs the number of misclassified samples. For others, it displays epochs vs costs.
        
        x(m, 2): m training samples with two features, x1 and x2 only.
                 Its shape is (m, 2); X0 must be set to False.
        x(m, 3): m training samples with two features x0=1, x1, x2
                  its shape is (m, 3); X0 must be set to True.
        y(m): m number of class labels, each value may be either 1 or -1,  
              also it may be either 1 or 0
   
        clf: an instance of classifiers (Perceptron, AdalineGD, AdalineSGD)
              
        X0: X has x_0 = 1 term in all samples or not; if True, removed before plotting
        annotate: add a sequence number at each sample if True
        savefig: save the plot in a file if a filename is given
        xylabels: two features of training data 
    """
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    if isinstance(clf, Perceptron):
        title = 'Perceptron'
    elif isinstance(clf, AdalineGD):
        title = 'AdalineGD'
    elif isinstance(clf, AdalineSGD):
        title = 'AdalineSGD'
    else:
        title = 'Unknown Classifier'

    if X0 == True:      # remove the first column; change shape(6x3) into shape(6x2)
        x = x[ : , 1:]  # check a column?: np.all(X == X[0,:], 
                        # axis = 0)[0] == True and X[0,0] == 1.0
                           
    # setting min max range of data - 10% of margin allowed in four sides
    rmin, rmax = x.min(), x.max()
    rmin -= (rmax - rmin) * 0.1    
    rmax +=(rmax - rmin) * 0.1   
    
    nums = [' {}'.format(i+1) for i in range(len(y))]    # numbering dots
    
    #y = y.squeeze()
    for num, ix, iy in zip(nums, x, y):
        if annotate == True:
            plt.annotate(num, xy=ix) 

    # This handles class 1 and -1, class 1 and 0 as well.
    ax[0].scatter(x[y==1, 0], x[y==1, 1], label=classes[0], marker='s', s=9)  
    ax[0].scatter(x[y!=1, 0], x[y!=1, 1], label=classes[1], marker='o', s=9)

    if clf is not None:
        w = clf.w[:]
        x1 = np.arange(rmin, rmax, .1)
        x2 = -w[0]/w[2] - w[1]/w[2]*x1 
        ax[0].plot(x1, x2)    
        title += ':w{}'.format(np.round(w, 2))          #display the weights at title

    ax[0].axhline(0, linewidth=1, linestyle='dotted')
    ax[0].axvline(0, linewidth=1, linestyle='dotted')
    ax[0].set_xlim([rmin, rmax])
    ax[0].set_ylim([rmin, rmax])
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_xlabel(xylabels[0])
    ax[0].set_ylabel(xylabels[1])
    ax[0].legend()        
    ax[0].set_title(title)
        
    if isinstance(clf, Perceptron):
        ax[1].plot(range(1, len(clf.cost_) + 1), clf.cost_, marker='o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Misclassified')
    elif isinstance(clf, (AdalineGD, AdalineSGD)):
        ax[1].plot(range(1, len(clf.cost_) + 1), clf.cost_, marker='o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Average Cost')
    if isinstance(clf, (Perceptron, AdalineGD, AdalineSGD)):
        ax[1].set_title('epochs:{}, eta:{}'.format(clf.epochs, clf.eta))

    if savefig is not None:
        fig.savefig(savefig, dpi=150)

        
def plot_xyw(x, y, W=None, X0=False, title='Perceptron', 
             classes=['class1', 'class2'], annotate=False, savefig=None):
    
    """ plots data x and its class label y as well as the the linear decision boundary
        and and the value W[-1] or w. 
        
        x(m, 2): m training samples with two features, x1 and x2 only.
                 Its shape is (m, 2); X0 must be set to False.
        x(m, 3): m training samples with two features x0=1, x1, x2
                  its shape is (m, 3); X0 must be set to True.
        y(m): m number of class labels, each value may be either 1 or -1,  
              also it may be either 1 or 0
   
        W(3,): only one boundary to display
               If you have an array of w's, but want to plot the last one, pass W[-1].
        W(epochs, 3): epochs number of decision boundaries or weights
              If there is one set of weights, its shape can be either (3, ) or (1, 3)
              
        X0: X has x_0 = 1 term in all samples or not; if True, removed before plotting
        annotate: add a sequence number at each sample if True
        savefig: save the plot in a file if a filename is given
        
    """
    if X0 == True:      # remove the first column; change shape(6x3) into shape(6x2)
        x = x[ : , 1:]  # check a column?: np.all(X == X[0,:], 
                        # axis = 0)[0] == True and X[0,0] == 1.0
    # print('x={}, y={}'.format(x, y))
                           
    # setting min max range of data - 10% of margin allowed in four sides
    rmin, rmax = x.min(), x.max()
    rmin -= (rmax - rmin) * 0.1    
    rmax +=(rmax - rmin) * 0.1   
    
    nums = ['  {}'.format(i+1) for i in range(len(y))]    # numbering dots
    
    #y = y.squeeze()
    for num, ix, iy in zip(nums, x, y):
        if annotate == True:
            plt.annotate(num, xy=ix) 
        ## can be replaced using plt.scatter
        ##if iy == 1: 
        ##    c1, = plt.plot(ix[0], ix[1], label='class 1', marker='s', color='blue')
        ##else:
        ##    c2, = plt.plot(ix[0], ix[1], label='class 2', marker='o', color='orange')

    # This handles class 1 and -1, class 1 and 0 as well.
    plt.scatter(x[y==1, 0], x[y==1, 1], label=classes[0], marker='s', s=9)  
    plt.scatter(x[y!=1, 0], x[y!=1, 1], label=classes[1], marker='o', s=9)

    if W is not None:
        if W.ndim == 1:                             # one boundary in1-d array shape(3,)
            x1 = np.arange(rmin, rmax, .1)
            x2 = -W[0]/W[2] - W[1]/W[2]*x1 
            plt.plot(x1, x2)    
            title += ':w{}'.format(np.round(W, 2))          #display the weights at title
        else: 
            for w in W:                                     # for every decision boundary
                x1 = np.arange(rmin, rmax, .1)
                x2 = -w[0]/w[2] - w[1]/w[2]*x1      
                #display all decision boundaries and legend-weights
                plt.plot(x1, x2, label='w:{}'.format(np.round(w, 2))) 
            title += ':w{}'.format(np.round(W[-1], 2))     #display the last weights at title

    plt.axhline(0, linewidth=1, linestyle='dotted')
    plt.axvline(0, linewidth=1, linestyle='dotted')
    plt.xlim([rmin, rmax])
    plt.ylim([rmin, rmax])
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)
    if W is not None and W.ndim != 1:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  
    else:
        plt.legend(loc='best')    
    plt.tight_layout()
    ##plt.legend([c1, c2], ["class 1", "class 2"])     # can be replaced by the line above
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', dpi=150)
        
        
        
    
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, clf_predict, resolution=0.01, scaled = True):
    # X, y: input examples and their class labels
    # clf or predict:  if clf is passed, then clf.predict() is used 
    #        predict method that returns the class label for examples
    # scaled:  use the same scale for x, y
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # plot the decision surface; the same ranges of x, y
    if scaled:
        rmin, rmax = X.min(), X.max()
        rmin -= (rmax - rmin) * 0.1    
        rmax += (rmax - rmin) * 0.1 
        x1_min = x2_min = rmin
        x1_max = x2_max = rmax

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    if callable(clf_predict):  # callable, use as it is
        Z = clf_predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    else:                        # not callable: invoke predict method of the classifier
        Z = clf_predict.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    #print("ending classifier.predict()")
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    #plt.xlim(xx1.min(), xx1.max())
    #plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    # print('X.shape={}, y.shape={}'.format(X.shape, y.shape))
    y = y.squeeze()        # for flexibility 
    for idx, cl in enumerate(np.unique(y)):
        # print('class={}, plot(x,y), x={}, y={}'.format(cl, X[y == cl, 0], X[y == cl, 1]))
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    
    # Added by idebtor@gmail.com
    # The following lines are added to make it square & clearer. 
    plt.axhline(0, linewidth=1, linestyle='dotted')
    plt.axvline(0, linewidth=1, linestyle='dotted')
    plt.xlim([x1_min, x1_max])
    plt.ylim([x2_min, x2_max])
    plt.gca().set_aspect('equal')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()      

def plot_decision_boundary(X, y, predict):
    """
    The original source comes from Andrew Ng's ML course on Coursera.
    The order of parameters are switched for compatibility with others in Joy.
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('$x_2$', fontsize=16)
    plt.xlabel('$x_1$', fontsize=16)
    
    #plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    #modified by yskim
    #plt.scatter(X[0, :], X[1, :], c=y.reshape(y.shape[1]), cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=y.squeeze(), cmap=plt.cm.Spectral)


    

def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

def tanh_d(x):
    return (1 + tanh(x))*(1 - tanh(x))

def sigmoid(x): 
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(x)
    """
    x = np.clip(x, -500, 500)  
    return 1/(1 + np.exp((-x)))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(x, 0)

def relu_d(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def one_hot_encoding(y, n_y, modified=True):   
    """y -- labels for training sets
             numpy array of m_samples, contains the target class labels for 
             each training example.
             For example, y = [2, 1, 5, 9] -> 4 training samples, and the ith 
             sample has label y[i]
       n_y -- number of neurons in output layer
       modified -- if False, use 0 and 1 only; Use 0.01 and 0.99 instead of 
               0 and 1, respectively.
       
       returns one-hot-encoding vector by y, its shape (m, n_y)
             m is the number of output labels or len(y).
             For each row, the ith index will be "hot", or 1, to represent that 
             index being the label.
    """
    yhot =  np.eye(n_y)[np.array(y, dtype='int32').flatten()]  
    if modified:
        yhot[yhot == 0] = 0.01
        yhot[yhot == 1] = 0.99
    return yhot


def one_hot_decoding(yhot):
    """ decodes one-hot-encoding vector into 1-d numpy array
        yhot -- one hot encoded numpy array
        returns 
        1-d numpy array
    """
    return np.argmax(yhot, axis=1)


#%%writefile code/perceptron.py
def perceptron(X, y, w = None, eta=0.1, epochs=5, random_seed=1):
    """ classifies data as and produces the linear decision boundary
        X(m, 3): m training samples with two features x0=1, x1, x2
                  its shape is (m,3)
        y(m,): m class labels
        w(3,1): initial weights if set
        
        Outputs:
        w(3,1): a decision boundary
    """
    if w is None:
        np.random.seed(random_seed) 
        w = np.random.random((X.shape[1],1))
    maxlabel, minlabel = y.max(), y.min()
    mid = (maxlabel + minlabel)/2
    for _ in range(epochs): 
        for xi, yi in zip(X, y):
            xi = xi.reshape(w.shape)
            z = np.dot(w.T, xi)                           
            yhat = np.where(z >= mid, maxlabel, minlabel) 
            delta = eta * (yi - yhat) * xi                  
            w += delta   
    #print('xi{}, \t (y,yhat)=({},{}), \tdelta{}, \tw{}'.
    #format(xi, yi, yhat, delta,np.round(w,2)))
    #print('w={}, w.shape={}'.format(w, w.shape))
    return w
        

# Implementation of Rosenblatt's perceptron algorithm for classification.
# Author: Youngsup KIm, idebtor@gmail.com
# 2018.03.01 - Creation
# 2018.04.18 - works with plot_descision_region(), net_input() modified

class Perceptron(object):
    """Perceptron classifier: This implementation of the Perceptron expects 
    binary class labels in {0, 1}.
    
    Parameters
        eta : float (default: 0.1), Learning rate (between 0.0 and 1.0)
        epochs : int (default: 10), Number of passes over the training dataset.
            Prior to each epoch, the dataset is shuffled to prevent cycles.
        random_seed : int, Random state for initializing random weights and shuffling.
        
        X0: If True, then X must have X_0 = 1 in all samples.
                Set it Faslse, if X does not have X_0 
    
    Attributes
        w  : 1d-array, shape={n_features, }, Model weights after fitting. Includes bias
        w_ : 2d-array, shape={epochs, n_features}, Weights in every epoch
        cost_ : list, Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.1, epochs=10, random_seed=1):
        #print("__init__ Perceptron")
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed

    def fit(self, X, y, X0=False):
        if X0 == False:                         #(m,3) = (m,) + (m, 2)
            X = np.c_[ np.ones(len(X)), X]      # padding x_0=1 for all samples
        
        np.random.seed(self.random_seed)
        self.w = np.random.random(X.shape[1]) 
            
        self.maxy, self.miny = y.max(), y.min()
        self.cost_ = []
        self.w_ = np.array([self.w])  
        
        for i in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                yhat = self.activate(xi)
                #print('xi{}, \t (yi,yhat)=({},{})'.format(np.round(xi,2), yi, yhat))
                if yi != yhat:
                    delta = self.eta * (yi - yhat) * xi    
                    self.w = self.w + delta 
                    #print('xi{}, \t (yi,yhat)=({},{}), delta{}, w{}'.format
                    #(np.round(xi,2), yi, yhat, np.round(delta,2), np.round(self.w,2)))
                    errors += 1
            self.cost_.append(errors)
            self.w_ = np.vstack([self.w_, self.w])
        return self

    def net_input(self, X):            # get the value of z
        #print('net_input X.shape={}, self.w={}'.format(X.shape, self.w))
        if X.shape[0] == self.w.shape[0]:   # used with X0 = True data 
            z = np.dot(X, self.w)
        else:     # plot_descision_region() invokes it with X w/o X0 
            z = np.dot(X, self.w[1:]) + self.w[0]    
        return z
    
    def activate(self, X):
        mid = (self.maxy + self.miny) / 2
        Z = self.net_input(X)
        return np.where(Z > mid, self.maxy, self.miny)
    
    def predict(self, X):   # plot_descision_region() invokes it with X w/o X0 
        return self.activate(X)

    
    

# Implementation of  Widrow's Adaptive Linear classifier algorithm
# Author: Youngsup KIm, idebtor@gmail.com
# 2018.03.21 - Creation
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.
    Parameters
        eta: float, Learning rate (between 0.0 and 1.0)
        epochs: int, Passes over the training dataset.
        random_seed : int, Random number generator seed for reproducibility

    Attributes
        w_ : 1d-array, Weights after fitting.
        cost_ : list, Sum-of-squares cost function value in each epoch.
    """
    def __init__(self, eta=0.01, epochs=10, random_seed=1):
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed

    def fit(self, X, y):
        """ Fit training data.
        Parameters
            X: numpy.ndarray, shape=(n_samples, m_features), 
            y: class label, array-like, shape = (n_samples, )
        Returns
            self : object
        """

        np.random.seed(self.random_seed)
        self.w = np.random.random(size=X.shape[1] + 1)

        self.maxy, self.miny = y.max(), y.min()
        self.cost_ = []
        self.w_ = np.array([self.w])
        
        #print('in fit(): X.shape{}, y.shape{}, self.w.shape{}'.
        #        format(X.shape, y.shape, self.w.shape))

        for i in range(self.epochs):
            Z = self.net_input(X)
            yhat = self.activation(Z)
            errors = (y - yhat)
            self.w[1:] += self.eta * np.dot(errors, X)
            self.w[0] += self.eta * np.sum(errors)
            cost = 0.5 * np.sum(errors**2)
            self.cost_.append(cost)
            self.w_ = np.vstack([self.w_, self.w]) 
        return self

    def net_input(self, X):            
        """Compute the value of z, net input  """
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation(self, X):  
        """Identity activation function: """
        return X

    def predict(self, X):      
        """Predict the class label with  """
        mid = (self.maxy + self.miny) / 2
        Z = self.net_input(X)
        yhat = self.activation(Z)
        return np.where(yhat > mid, self.maxy, self.miny)
    
    
    
    
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.
    Parameters
        eta : float, Learning rate (between 0.0 and 1.0)
        epochs : int, Passes over the training dataset.
    Attributes
        w  : 1d-array, Weights after fitting.
        w_ : 2d-array, Array of weights w accumulated every epoch 
        cost_ : list, Sum-of-squares cost function value averaged over all
            training samples in each epoch.
    shuffle : bool (default: True), Shuffles training data every epoch 
            if True to prevent cycles.
    random_seed : int (default: None)
        Set random seed for shuffling and initializing the weights.
    """
    def __init__(self, eta=0.01, epochs=10, shuffle=True, random_seed=None):
        self.eta = eta
        self.epochs = epochs
        self.w_initialized = False
        self.shuffle = shuffle
        if random_seed:
            np.random.seed(random_seed)

    def fit(self, X, y):
        """ Fit training data.
        Parameters
            X: numpy.ndarray, shape=(n_samples, m_features), 
            y: class label, array-like, shape = (n_samples, )
        Returns
        -------
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        self.w_ = np.array([self.w])
        self.maxy, self.miny = y.max(), y.min()
        
        for i in range(self.epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
            self.w_ = np.vstack([self.w_, self.w]) 
            
        #print('fit X.shape={}, X.ndim={}, self.w={}'.format(X.shape, X.ndim, self.w))
        
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, y):
        """Apply Adaline learning rule to update the weights"""
        yhat = self.net_input(xi)
        error = (y - yhat)
        self.w[1:] += self.eta * xi.dot(error)
        self.w[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):   
        """Calculate net input"""       
        z = np.dot(X, self.w[1:]) + self.w[0]    
        return z

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        #print('predict X.shape={}, X.ndim={}, self.w={}'.format(X.shape, X.ndim, self.w))
        mid = (self.maxy + self.miny) / 2
        return np.where(self.activation(self.net_input(X)) > mid, self.maxy, self.miny)
    
    

class MnistBGD(object):
    """ Batch Gradient Descent for MNIST Dataset """
    def __init__(self, n_x, n_h, n_y, eta = 0.1, epochs = 100, random_seed=1):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.eta = eta
        self.epochs = epochs
        np.random.seed(random_seed)
        self.W1 = 2*np.random.random((self.n_h, self.n_x)) - 1  # between -1 and 1
        self.W2 = 2*np.random.random((self.n_y, self.n_h)) - 1  # between -1 and 1
        
    def forpass(self, A0):
        Z1 = np.dot(self.W1, A0)    # hidden layer inputs
        A1 = self.g(Z1)             # hidden layer outputs/activation func
        Z2 = np.dot(self.W2, A1)    # output layer inputs
        A2 = self.g(Z2)             # output layer outputs/activation func
        return Z1, A1, Z2, A2

    def fit(self, X, y):
        self.m_samples = len(y)       
        Y = one_hot_encoding(y, self.n_y)  
        
        self.cost_ = []
        for epoch in range(self.epochs):
            A0 = np.array(X, ndmin=2).T     
            Y0 = np.array(Y, ndmin=2).T      

            Z1 = np.dot(self.W1, A0)          
            A1 = self.g(Z1)                  
            Z2 = np.dot(self.W2, A1)       
            A2 = self.g(Z2)                 

            E2 = Y0 - A2                   
            E1 = np.dot(self.W2.T, E2)          

            dZ2 = E2 * self.g_prime(Z2)       
            dZ1 = E1 * self.g_prime(Z1)  
            
            dW2 = self.eta * np.dot(dZ2, A1.T)
            dW1 = self.eta * np.dot(dZ1, A0.T)

            self.W2 += dW2 / self.m_samples   
            self.W1 += dW1 / self.m_samples    

            self.cost_.append(np.sqrt(np.sum(E2 * E2)))
        return self

    def predict(self, X):
        A0 = np.array(X, ndmin=2).T         # A0: inputs
        Z1, A1, Z2, A2 = self.forpass(A0)   # forpass
        return A2                                       

    def g(self, x):                 # activation_function: sigmoid
        x = np.clip(x, -500, 500)   # prevent from overflow, 
        return 1.0/(1.0+np.exp(-x)) # stackoverflow.com/questions/23128401/
                                    # overflow-error-in-neural-networks-implementation
    
    def g_prime(self, x):           # activation_function: sigmoid derivative
        return self.g(x) * (1 - self.g(x))
    
    def evaluate(self, Xtest, ytest):      
        m_samples = len(ytest)
        scores = 0        
        A2 = self.predict(Xtest)
        print(A2.shape)
        yhat = np.argmax(A2, axis = 0)
        scores += np.sum(yhat == ytest)
        return scores/m_samples * 100

 
class MnistBGD_LS(object):
    """ Batch Gradient Descent with a simple learning schedule   """
    def __init__(self, n_x, n_h, n_y, eta = 0.1, epochs = 100, random_seed=1):
        """ 
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.eta = eta
        self.epochs = epochs
        np.random.seed(random_seed)
        self.W1 = 2*np.random.random((self.n_h, self.n_x)) - 1  # between -1 and 1
        self.W2 = 2*np.random.random((self.n_y, self.n_h)) - 1  # between -1 and 1
        
    def forpass(self, A0):
        Z1 = np.dot(self.W1, A0)          # hidden layer inputs
        A1 = self.g(Z1)                      # hidden layer outputs/activation func
        Z2 = np.dot(self.W2, A1)          # output layer inputs
        A2 = self.g(Z2)                       # output layer outputs/activation func
        return Z1, A1, Z2, A2

    def fit(self, X, y):
        self.cost_ = []
        self.m_samples = len(y)       
        Y = one_hot_encoding(y, self.n_y)     
        # learning rate is scheduled to decrement by a step of 
        # which the inteveral from self.eta to 0.0001 eqaully 
        # divided by total number of iterations(epochs or 
        # epochs * m_samples)
        eta_scheduled = np.linspace(self.eta, 0.0001, self.epochs)
        for epoch in range(self.epochs):
            A0 = np.array(X, ndmin=2).T       
            Y0 = np.array(Y, ndmin=2).T     

            Z1, A1, Z2, A2 = self.forpass(A0)  
            E2 = Y0 - A2                      
            E1 = np.dot(self.W2.T, E2)         

            dZ2 = E2 * self.g_prime(Z2)          
            dZ1 = E1 * self.g_prime(Z1)        

            eta = eta_scheduled[epoch]
            self.W2 +=  eta * np.dot(dZ2, A1.T) / self.m_samples    
            self.W1 +=  eta * np.dot(dZ1, A0.T) / self.m_samples    
            self.cost_.append(np.sqrt(np.sum(E2 * E2)))
        return self

    def predict(self, X):
        A0 = np.array(X, ndmin=2).T         # A0: inputs
        Z1, A1, Z2, A2 = self.forpass(A0)   # forpass
        return A2                                       

    def g(self, x):                 # activation_function: sigmoid
        x = np.clip(x, -500, 500)   # prevent from overflow, 
        return 1.0/(1.0+np.exp(-x)) # stackoverflow.com/questions/23128401/
                                    # overflow-error-in-neural-networks-implementation
    
    def g_prime(self, x):                    # activation_function: sigmoid derivative
        return self.g(x) * (1 - self.g(x))
    
    def evaluate(self, Xtest, ytest):       # fully vectorized calculation
        m_samples = len(ytest)
        scores = 0        
        A2 = self.predict(Xtest)
        yhat = np.argmax(A2, axis = 0)
        scores += np.sum(yhat == ytest)
        return scores/m_samples * 100
    
    
class MnistSGD(object):
    """
    MNIST 데이터셋을 확률적 경사하강법으로 0 ~ 9로 분류함. 
    """
    def __init__(self, n_x, n_h, n_y, eta = 0.1, epochs = 1, random_seed=1):
        """
        클래스 객체를 만들기 위한 초기화 설정
        n_x -- 입력층 노드의 수
        n_y -- 은닉층 노드의 수
        n_y -- 출력층 노드의 수
        eta -- 학습률
        epochs -- 훈련 반복 횟수
        random_seed -- 난수 발생의 규칙적 반복을 위하여 설정함
        W1, W2 -- 입력층과 은닉층 및 은닉층과 출력층 사이의 가중치
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.eta = eta
        self.epochs = epochs
        np.random.seed(random_seed)
        self.W1 = 2*np.random.random((self.n_h, self.n_x)) - 1  # between -1 and 1
        self.W2 = 2*np.random.random((self.n_y, self.n_h)) - 1  # between -1 and 1
        
    def forpass(self, A0):
        '''
        순전파를 계산함.
        A0 -- 입력층의 입력
        반환 
        Z1, A1 -- 은닉층의 입력과 출력
        Z2, A2 -- 출력층의 입력과 출력
        '''
        Z1 = np.dot(self.W1, A0)          # hidden layer inputs
        A1 = self.g(Z1)                      # hidden layer outputs/activation func
        Z2 = np.dot(self.W2, A1)          # output layer inputs
        A2 = self.g(Z2)                       # output layer outputs/activation func
        return Z1, A1, Z2, A2

    def fit(self, X, y):
        """ 
        순전파/역전파를 반복적으로 실행하며, 가중치를 경사하강법으로 조정함
        X -- 입력 이미지 (m, 784)
        y -- 입력 레이블 (m, )
        """

        self.cost_ = []
        m_samples = len(y)
        Y = one_hot_encoding(y, self.n_y)       # (m, n_y) = (m, 10)   one-hot encoding
        
        for epoch in range(self.epochs):
            print('Training epoch {}/{}.'.format(epoch+1, self.epochs))
            
            for m in range(m_samples):            
                # input X can be tuple, list, or ndarray
                A0 = np.array(X[m], ndmin=2).T     # A0 : inputs, minimum 2 dimensional array
                Y0 = np.array(Y[m], ndmin=2).T       # Y: targets

                Z1, A1, Z2, A2 = self.forpass(A0)          # forward pass
                if epoch == 0 and m == 0:
                    print('A0.shape={}, Y0.shape={}'.format(A0.shape, Y0.shape))
                    print('Z1.shape={}, A1.shape={}, Z2.shape={}, A2.shape={}'.
                          format(Z1.shape, A1.shape, Z2.shape, A2.shape))

                E2 = Y0 - A2                       # E2: output errors
                E1 = np.dot(self.W2.T, E2)         # E1: hidden errors

                # back prop and update weights
                #self.W2 += self.eta * np.dot(E2 * A2 * (1.0 - A2), A1.T)
                #self.W1 += self.eta * np.dot(E1 * A1 * (1.0 - A1), A0.T)

                # back prop, error prop
                dZ2 = E2 * self.g_prime(Z2)        # backprop      # dZ2 = E2 * A2 * (1 - A2)  
                dZ1 = E1 * self.g_prime(Z1)        # backprop      # dZ1 = E1 * A1 * (1 - A1)  
                if epoch == 0 and m == 0:
                    print('E1.shape={}, E2.shape={}, dZ1.shape={}, dZ2.shape={}'.
                          format(E1.shape, E2.shape, dZ1.shape, dZ2.shape))

                # update weights
                self.W2 +=  self.eta * np.dot(dZ2, A1.T)      # update output layer weights
                self.W1 +=  self.eta * np.dot(dZ1, A0.T)      # update hidden layer weights
                self.cost_.append(np.sqrt(np.sum(E2 * E2)))
        return self

    def predict(self, X):
        A0 = np.array(X, ndmin=2).T         # A0: inputs
        Z1, A1, Z2, A2 = self.forpass(A0)   # forpass
        return A2                                       

    def g(self, x):                             # activation_function: sigmoid
        return 1.0/(1.0+np.exp(-x))
    
    def g_prime(self, x):                    # activation_function: sigmoid derivative
        return self.g(x) * (1 - self.g(x))
    
    def evaluate(self, Xtest, ytest):       # fully vectorized calculation
        m_samples = len(ytest)
        scores = 0        
        A2 = self.predict(Xtest)
        yhat = np.argmax(A2, axis = 0)
        scores += np.sum(yhat == ytest)
        return scores/m_samples * 100
        
    def evaluate_onebyone(self, Xtest, ytest):
        m_samples = len(ytest)
        scores = 0
        for m in range(m_samples):
            A2 = self.predict(Xtest[m])
            yhat = np.argmax(A2)
            if yhat == ytest[m]:
                scores += 1
        return scores/m_samples * 100
    



class MnistMiniBatchGD(object):
    """ Mini-batch Gradient Descent
    """
    def __init__(self, n_x, n_h, n_y, eta = 0.1, epochs = 100, batch_size = 32, random_seed=1):
        """ 
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        np.random.seed(random_seed)
        self.W1 = 2*np.random.random((self.n_h, self.n_x)) - 1  # between -1 and 1
        self.W2 = 2*np.random.random((self.n_y, self.n_h)) - 1  # between -1 and 1
        #print('W1.shape={}, W2.shape={}'.format(self.W1.shape, self.W2.shape))
        
    def forpass(self, A0, train=True):
        Z1 = np.dot(self.W1, A0)          # hidden layer inputs
        A1 = self.g(Z1)                      # hidden layer outputs/activation func
        '''
        # Dropout
        if train:
            self.drop_units = np.random.rand(*A1.shape) > self.dropout_ratio
            A1 = A1 * self.drop_units
        else:
            A1 = A1 * (1.0 - self.dropout_ratio)
        '''
           
        Z2 = np.dot(self.W2, A1)          # output layer inputs
        A2 = self.g(Z2)                       # output layer outputs/activation func
        return Z1, A1, Z2, A2

    def fit(self, X, y):
        """ 
        X: input dataset 
        y: class labels
        """

        self.cost_ = []
        m_samples = len(y)       
        Y = one_hot_encoding(y, self.n_y)       # (m, n_y) = (m, 10)   one-hot encoding
        #print('X.shape={}, y.shape={}, Y.shape={}'.format(X.shape, y.shape, Y.shape))
        
        for epoch in range(self.epochs):
            print('Training epoch {}/{}.'.format(epoch + 1, self.epochs))
            for i in range(0, m_samples, self.batch_size):
                A0 = X[i: i + self.batch_size]
                Y0 = Y[i: i + self.batch_size]
                
                A0 = np.array(A0, ndmin=2).T
                Y0 = np.array(Y0, ndmin=2).T

                Z1, A1, Z2, A2 = self.forpass(A0)        

                E2 = Y0 - A2                 
                E1 = np.dot(self.W2.T, E2)       

                # back prop, error prop
                dZ2 = E2 * self.g_prime(Z2)     
                dZ1 = E1 * self.g_prime(Z1)    
                '''
                # Dropout
                dZ1 = dZ1 * self.drop_units       
                '''
                
                # update weights
                self.W2 +=  self.eta * np.dot(dZ2, A1.T)     
                self.W1 +=  self.eta * np.dot(dZ1, A0.T)    

                self.cost_.append(np.sqrt(np.sum(E2 * E2)
                                          /self.batch_size))
        return self

    def predict(self, X):
        A0 = np.array(X, ndmin=2).T         # A0: inputs
        Z1, A1, Z2, A2 = self.forpass(A0, train=False)   # forpass
        return A2                                       

    def g(self, x):                             # activation_function: sigmoid
        return 1.0/(1.0+np.exp(-x))
    
    def g_prime(self, x):                    # activation_function: sigmoid derivative
        return self.g(x) * (1 - self.g(x))
    
    def evaluate(self, Xtest, ytest):       
        m_samples = len(ytest)
        scores = 0        
        A2 = self.predict(Xtest)
        yhat = np.argmax(A2, axis = 0)
        scores += np.sum(yhat == ytest)
        return scores/m_samples * 100    
    
    
    
class NeuralNetwork():
    """ This class implements a multi-perceptron with backpropagation. """
    def __init__(self, net_arch, eta=0.1, epochs=100, random_seed=1):
        self.layers = len(net_arch)
        self.net_arch = net_arch
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed
        
    def fit(self, X, Y):
        """ 
        X: input dataset in row vector style, 
        Y: class labels
        w: optional weights, its shape is (3, 1)
        """
        # seed random numbers to make calculation deterministic 
        # initialize weights randomly with mean 0
        np.random.seed(self.random_seed)
        self.W1 = 2*np.random.random((self.net_arch[1], self.net_arch[0])) - 1
        self.W2 = 2*np.random.random((self.net_arch[2], self.net_arch[1])) - 1      
        #print('X.shape={}, Y.shape{}'.format(X.shape, Y.shape))
        #print('W1.shape={}, W2.shape={}'.format(self.W1.shape, self.W2.shape))

        self.cost_ = []
        
        for iter in range(self.epochs):
            A0 = X                             # unnecessary, but to illustrate only
            Z1 = np.dot(self.W1, A0)           # hidden layer input
            A1 = self.g(Z1)                    # hidden layer output
            Z2 = np.dot(self.W2, A1)           # output layer input
            A2 = self.g(Z2)                    # output layer results

            E2 = Y - A2                        # error @ output
            E1 = np.dot(self.W2.T, E2)         # error @ hidden

            # multiply the error by the sigmoid slope at the values in Z? or A?
            dZ2 = self.eta * E2 * self.g_prime(Z2)        # backprop      # dZ2 = E2 * A2 * (1 - A2)  
            dZ1 = self.eta * E1 * self.g_prime(Z1)        # backprop      # dZ1 = E1 * A1 * (1 - A1)  

            self.W2 +=  np.dot(dZ2, A1.T)      # update output layer weights
            self.W1 +=  np.dot(dZ1, A0.T)      # update hidden layer weights
            self.cost_.append(np.sqrt(np.sum(E2 * E2)))
        return self

    def net_input(self, X):                     ## sum-product  z
        if X.shape[0] == self.w.shape[0]:   # used with X0 = True data 
            return np.dot(X, self.w)
        else:
            return np.dot(X, self.w[1:]) + self.w[0]
    
    def g(self, x):    # activation function
        return 1/(1 + np.exp((-x)))
    
    def g_prime(self, x):  # gradient or sigmoid derivative
        return self.g(x) * (1 - self.g(x))

    def predict(self, X): 
        #print('predict: W1.shape:{}, Xshape:{} '.format(self.W1.shape, X.shape))
        Z1 = np.dot(self.W1, X.T)           # hidden layer input
        A1 = self.g(Z1)                     # hidden layer output
        Z2 = np.dot(self.W2, A1)            # output layer input
        A2 = self.g(Z2)                     # output layer results
        return A2
   


class LogisticNeuron(object):
    """ implements logistic regression using cross entropy 
    Arguments:
    X -- dataset of shape (2, number of examples) or (n_x, m)
    Y -- "true" labels vector of shape (1, number of examples)
    n_x -- X.shape[0]
    n_h -- size of the hidden layer
    n_y -- Y.shape[0]
    eta -- learning rate
    epochs -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Saved: parameters learnt by the model. Some can can be used to predict.
    W1, W2, b1, b2 -- weights and bias
    cost_ -- a list of error in every epoch saved
    """
    def __init__(self, n_h, eta=0.2, epochs=1000, random_seed=1, print_cost=False):
        self.n_h = n_h         # size of hidden layer
        self.eta = eta         # learning rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.print_cost = print_cost
        
    # Define the cost function A2, y
    def CEcost(self, A2, Y):
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        cost = - np.sum(logprobs) / self.m_samples
        cost = np.squeeze(cost) 
        return cost   # makes sure cost is the dimension we expect. 
                      # E.g., turns [[17]] into 17 
 
    def forpass(self, X):
        """ 순전파 Z=WX, A=g(Z)를 계산하고 각 노드의 결과값 A1, A2를 반환합니다. """
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)                  # yhat
        assert(A2.shape == (1, X.shape[1]))
        return A1, A2 
            
    def fit(self, X, Y):
        """순전파를 계산하고 역전파를 이용하여 모델을 최적화합니다"""
        self.m_samples = Y.shape[1] # number of example
        self.n_x = X.shape[0] # size of input layer
        self.n_y = Y.shape[0] # size of output layer
        # seed random numbers to make calculation deterministic 
        # initialize weights randomly with mean 0
        np.random.seed(self.random_seed)
        self.W1 = 2*np.random.random((self.n_h, self.n_x)) - 1
        self.b1 = np.zeros((self.n_h, 1))
        self.W2 = 2*np.random.random((self.n_y, self.n_h)) - 1      
        self.b2 = np.zeros((self.n_y, 1))
        #print('X.shape={}, Y.shape{}'.format(X.shape, Y.shape))
        #print('W1.shape={}, W2.shape={}'.format(self.W1.shape, self.W2.shape))
        assert(X.shape[1] == Y.shape[1])
        self.cost_ = []
        
        # Loop (gradient descent)
        for i in range(0, self.epochs):
            # Forward propagation. 
            A0 = X
            A1, A2 = self.forpass(X)
            
            # Cost function: Compute the cross-entropy cost
            cost = self.CEcost(A2, Y)
            self.cost_.append(cost)
                        
            # Backpropagation. 
            E2 = Y - A2
            dZ2 = E2
            dW2 = np.dot(dZ2, A1.T)/self.m_samples 
            db2 = np.sum(dZ2, axis=1, keepdims=True)/self.m_samples
            
            E1 = np.dot(self.W2.T, E2)
            dZ1 = E1 * (1 - np.power(A1, 2))
            dW1 = np.dot(dZ1, A0.T)/self.m_samples
            db1 = np.sum(dZ1, axis=1, keepdims=True)/self.m_samples
            
            # Gradient descent parameter update.           
            self.W1 += self.eta * dW1
            self.b1 += self.eta * db1
            self.W2 += self.eta * dW2
            self.b2 += self.eta * db2

            # Print the cost every 1000 iterations
            if self.print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        return self
            
    def predict(self, X): 
        """이미 학습된 모델로 입력 X에 대한 예측값을 반환합니다.
        Arguments:
        X -- input data of size (n_x, m)
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        # Computes probabilities using forward propagation, and 
        # classifies to 0/1 using 0.5 as the threshold.
        A1, A2 = self.forpass(X)       
        assert(A2.shape == (1, X.shape[1]))
        predictions = A2 > 0.5                    # returns true or false 
        return predictions



class LogisticNeuron_stochastic(object):
    """implements Logistic Regression using cross entropy with stochastic gradient descent"""
    def __init__(self, n_x, n_h, n_y, eta = 0.2, epochs = 5, random_seed=1):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed
        
        np.random.seed(self.random_seed)
        self.W1 = 2*np.random.random((self.n_h, self.n_x)) - 1
        self.b1 = np.zeros((self.n_h, 1))
        self.W2 = 2*np.random.random((self.n_y, self.n_h)) - 1      
        self.b2 = np.zeros((self.n_y, 1))
        self.W1 = 2*np.random.random((self.n_h, self.n_x)) - 1  
        self.W2 = 2*np.random.random((self.n_y, self.n_h)) - 1  
        
    def CEcost(self, A2, Y):
        m = Y.shape[1]      # number of example
        logprobs = np.multiply(Y, np.log(A2))
        cost = -np.sum(logprobs)/m
        cost = np.squeeze(cost)        
        return cost  
    
    def forpass(self, A0):
        Z1 = np.dot(self.W1, A0) + self.b1         
        A1 = self.g(Z1)                          
        Z2 = np.dot(self.W2, A1) + self.b2       
        A2 = self.softmax(Z2)                   
        return Z1, A1, Z2, A2

    def fit(self, X, y): 
        self.cost_ = []
        self.m_samples = len(y)
        Y = one_hot_encoding(y, self.n_y)       # (m, n_y) = (m, 10)   one-hot encoding
               
        for epoch in range(self.epochs):           
            for sample in range(self.m_samples):            
                A0 = np.array(X[sample], ndmin=2).T  
                Y0 = np.array(Y[sample], ndmin=2).T  

                Z1, A1, Z2, A2 = self.forpass(A0)          # forward pass
                
                # Cost function: Compute the cross-entropy cost
                cost = self.CEcost(A2, Y0)
                self.cost_.append(cost)
                # Backpropagation. 
                E2 = Y0 - A2                
                dZ2 = E2 
                dW2 = np.dot(dZ2, A1.T) / self.m_samples
                db2 = np.sum(dZ2, axis=1, keepdims=True) / self.m_samples
                
                E1 = np.dot(self.W2.T, E2)  
                dZ1 = E1 * self.g_prime(Z1)  #sigmoid
                #dZ1 = E1 * (1 - np.power(A1, 2)) #tanh
                dW1 = np.dot(dZ1, A0.T) 
                db1 = np.sum(dZ1, axis=1, keepdims=True) 
                
                # update weights 
                self.W1 += self.eta * dW1 
                self.b1 += self.eta * db1 
                self.W2 += self.eta * dW2 
                self.b2 += self.eta * db2 
            print('Training epoch {}/{}, cost = {}'.format(epoch+1, self.epochs, cost))
        return self

    def predict(self, X):
        A0 = np.array(X, ndmin=2).T         # A0: inputs
        Z1, A1, Z2, A2 = self.forpass(A0)   # forpass
        return A2  

    def g(self, x):                 # activation_function: sigmoid
        x = np.clip(x, -500, 500)   # prevent from overflow, 
        return 1.0/(1.0+np.exp(-x)) # stackoverflow.com/questions/23128401/
                                    # overflow-error-in-neural-networks-implementation
    
    def g_prime(self, x):           # activation_function: sigmoid derivative
        return self.g(x) * (1 - self.g(x))
    
    def softmax(self, a):           # prevent it from overlfow and undeflow
        exp_a = np.exp(a - np.max(a))
        return exp_a / np.sum(exp_a)
    
    def evaluate(self, Xtest, ytest):   # fully vectorized calculation
        m_samples = len(ytest)  
        A2 = self.predict(Xtest)
        yhat = np.argmax(A2, axis = 0)
        scores = np.sum(yhat == ytest)
        return scores/m_samples * 100
    
    
def load_cat_data():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


fashion_url_base = 'https://github.com/zalandoresearch/fashion-mnist'
fashion_key_file = {
    'train_img':'fashion-train-images-idx3-ubyte.gz',
    'train_label':'fashion-train-labels-idx1-ubyte.gz',
    'test_img':'fashion-t10k-images-idx3-ubyte.gz',
    'test_label':'fashion-t10k-labels-idx1-ubyte.gz'
}


def download_fashion_mnist():
    for v in fashion_key_file.values():
        _download(fashion_url_base, v)
        
def _fashion_convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(fashion_key_file['train_img'])
    dataset['train_label'] = _load_label(fashion_key_file['train_label'])    
    dataset['test_img'] = _load_img(fashion_key_file['test_img'])
    dataset['test_label'] = _load_label(fashion_key_file['test_label'])
    
    return dataset        

def init_fashion_mnist():
    download_fashion_mnist()
    dataset = _fashion_convert_numpy()
    print("Creating pickle file ...", end='')
    with open(fashion_save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def load_fashion_mnist(normalize=True, flatten=True):
    if not os.path.exists(fashion_save_file):
        init_fashion_mnist()
        
    with open(fashion_save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    """        
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    """
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 28, 28)  

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])



class DeepNeuralNet(object):
    """ implements a deep neural net. 
        Users may specify any number of layers.
        net_arch -- consists of a number of neurons in each layer 
    """
    def __init__(self, net_arch, activate = None, eta = 1.0, epochs = 100, random_seed = 1):
        self.eta = eta
        self.epochs = epochs
        self.net_arch = net_arch
        self.layers = len(net_arch)
        self.W = []
        self.random_seed = random_seed
        
        self.g       = [lambda x: sigmoid(x)   for _ in range(self.layers)]
        self.g_prime = [lambda x: sigmoid_d(x) for _ in range(self.layers)]
        
        if activate is not None:
            for i, (g, g_prime) in enumerate(zip(activate[::2], activate[1::2])):
                self.g[i+1] = g
                self.g_prime[i+1] = g_prime
                
        for i in range(len(self.g)):
            print(type(self.g[i]), id(self.g[i]))
        
        #print('X.shape={}, y.shape{}'.format(X.shape, y.shape))
        # Random initialization with range of weight values (-1,1)
        np.random.seed(self.random_seed)
        
        # A place holder [None] is used to indicated "unused place".
        self.W = [[None]]    ## the first W0 is not used.
        for layer in range(self.layers - 1):
            w = 2 * np.random.rand(self.net_arch[layer+1], 
                                   self.net_arch[layer]) - 1
            self.W.append(w)  
            
    def forpass(self, A0):     
        Z = [[None]]   # Z0 is not used.
        A = []       # A0 = X0 is used. 
        A.append(A0)
        for i in range(1, len(self.W)):
            z = np.dot(self.W[i], A[i-1])
            Z.append(z)
            a = self.g[i](z)
            A.append(a)
        return Z, A
    
    def backprop(self, Z, A, Y):
        # initialize empty lists to save E and dZ
        # A place holder None is used to indicated "unused place".
        E  = [None for x in range(self.layers)]
        dZ = [None for x in range(self.layers)]
        
        # Get error at the output layer or the last layer
        ll = self.layers - 1
        error = Y - A[ll]
        E[ll] = error   
        dZ[ll] = error * self.g_prime[ll](Z[ll]) 
        
        # Begin from the back, from the next to last layer
        for i in range(self.layers-2, 0, -1):
            E[i]  = np.dot(self.W[i+1].T, E[i+1])
            dZ[i] = E[i] * self.g_prime[i](Z[i])
       
        # Adjust the weights, using the backpropagation rules
        m = Y.shape[0] # number of samples
        for i in range(ll, 0, -1):
            self.W[i] += self.eta * np.dot(dZ[i], A[i-1].T) / m
        return error
         
    def fit(self, X, y):
        self.cost_ = []        
        for epoch in range(self.epochs):          
            Z, A = self.forpass(X)        
            cost = self.backprop(Z, A, y)   
            self.cost_.append(
                 np.sqrt(np.sum(cost * cost)))    
        return self

    def predict(self, X):
        A0 = np.array(X, ndmin=2).T         # A0: inputs
        Z, A = self.forpass(A0)     # forpass
        return A[-1]                                       
   
    def evaluate(self, Xtest, ytest):       # fully vectorized calculation
        m_samples = len(ytest)
        scores = 0        
        A3 = self.predict(Xtest)
        yhat = np.argmax(A3, axis = 0)
        scores += np.sum(yhat == ytest)
        return scores/m_samples * 100
    
    

class DeepNeuralNet_BGD(object):
    """ implements a deep neural net. Users may specify any number 
        of layers.
        net_arch -- consists of a number of neurons in each layer 
    """
    def __init__(self, net_arch, activate = None, eta = 1.0, epochs = 100, random_seed = 1):
        
        if not isinstance(net_arch, list):
            sys.exit('Use a list to list number of neurons in each layer.')
        if len(net_arch) < 3:
            sys.exit('Specify the number of neurons more than two layers.')
                     
        self.eta = eta
        self.epochs = epochs
        self.net_arch = net_arch
        self.layers = len(net_arch)
        self.W = []
        self.random_seed = random_seed
        
        np.random.seed(self.random_seed)
        # Random initialization with range of weight values (-1,1)
        # A place holder None is used to indicated "unused place".
        self.W = [[None]]    ## the first W0 is not used.
        for layer in range(self.layers - 1):
            w = 2 * np.random.rand(self.net_arch[layer+1], 
                                   self.net_arch[layer]) - 1
            self.W.append(w)
        
        # initialize the activation function list with sigmoid() as default
        self.g = [lambda x: sigmoid(x) for _ in range(self.layers)]
        self.g_prime = [lambda x: sigmoid_d(x) for _ in range(self.layers)]
        
        # get the user-defined activation functions and their derivatives
        if activate is not None:
            if len(activate) % 2 != 0:
                sys.exit("List activation functions & its derivatives in pairwise")
            if len(activate) > (self.layers - 1) * 2:
                sys.exit("Too many activation functions & its derivatives encountered")
            for i, (g, g_prime) in enumerate(zip(activate[::2], activate[1::2])):
                self.g[i+1] = g
                self.g_prime[i+1] = g_prime
            
    def forpass(self, A0):     
        Z = [[None]] # Z0 is not used.
        A = []       # A0 = X0 is used. 
        A.append(A0)
        for i in range(1, len(self.W)):
            z = np.dot(self.W[i], A[i-1])
            Z.append(z)
            a = self.g[i](z)
            A.append(a)
        return Z, A
    
    def backprop(self, Z, A, Y):
        # initialize empty lists to save E and dZ
        # A place holder None is used to indicated "unused place".
        E  = [None for x in range(self.layers)]
        dZ = [None for x in range(self.layers)]
        
        # Get error at the output layer or the last layer
        ll = self.layers - 1
        error = Y - A[ll]
        E[ll] = error   
        dZ[ll] = error * self.g_prime[ll](Z[ll]) 
        
        # Begin from the back, from the next to last layer
        for i in range(self.layers-2, 0, -1):
            E[i]  = np.dot(self.W[i+1].T, E[i+1])
            dZ[i] = E[i] * self.g_prime[i](Z[i])
       
        # Adjust the weights 
        m = Y.shape[1]  # number of samples
        for i in range(ll, 0, -1):
            self.W[i] += self.eta * np.dot(dZ[i], A[i-1].T) / m
        return error
         
    def fit(self, X, y):
        self.cost_ = [] 
        self.m_samples = len(y)
        Y = one_hot_encoding(y, self.net_arch[-1]) 
        
        for epoch in range(self.epochs): 
            #if epoch % 20== 0:
            #    print('Training epoch {}/{}'.format(epoch+1, self.epochs))

            A0 = np.array(X, ndmin=2).T   # A0 : inputs, minimum 2d array
            Y0 = np.array(Y, ndmin=2).T   # Y: targets

            Z, A = self.forpass(A0)          # forward pass
            cost = self.backprop(Z, A, Y0)   # back propagation
            self.cost_.append(np.sqrt(np.sum(cost * cost)))
        return self

    def predict(self, X):  # used in plot_decsion_regions()          
        Z, A2 = self.forpass(X)
        A2 = np.array(A2[len(A2)-1])
        return A2[-1] > 0.5
    
    def predict_(self, X): # used in evaluate() 
        A0 = np.array(X, ndmin=2).T         # A0: inputs
        Z, A = self.forpass(A0)             # forpass
        return A[-1]                                       
   
    def evaluate(self, Xtest, ytest):       # fully vectorized calculation
        m_samples = len(ytest)
        scores = 0        
        A3 = self.predict_(Xtest)
        yhat = np.argmax(A3, axis = 0)
        scores += np.sum(yhat == ytest)
        return scores/m_samples * 100