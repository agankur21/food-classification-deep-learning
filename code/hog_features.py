import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn import svm
from sklearn.cross_validation import train_test_split
import cPickle
from scipy import misc

def rgb2gray(rgb):
    """Convert RGB image to grayscale

      Parameters:
        rgb : RGB image

      Returns:
        gray : grayscale image

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image

         Modified from skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
        im : an input grayscale or rgb image

      Returns:
        feat: Histogram of Gradient (HOG) feature

    """

    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # image size
    orientations = 9  # number of gradient bins
    cx, cy = (8, 8)  # pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[cx / 2::cx, cy / 2::cy].T

    return orientation_histogram.ravel()

f1 = open('/mnt/data/train_data.txt', 'rb')
data_dictionary = {}
class_count = {}
lines = f1.readlines()
f1.close()
for l in lines:
    l = l.replace('\n', '')
    imgname, label = l.split(' ')
    imgname = '/mnt/data/train/' + imgname
    if label not in class_count:
        class_count[label] = 1
    elif class_count[label] >= 100:
        continue
    else:
        class_count[label] += 1
    data_dictionary[imgname] = int(label)
X = None
Y = []
count = 0
for imgname in data_dictionary:
    img = misc.imread(imgname)
    if X is None:
        X = hog_feature(img)
    else:
        X = np.vstack((X, hog_feature(img)))
    Y.append(data_dictionary[imgname])
    if count % 1000 == 0:
        print count
    count += 1

Y = np.array(Y)
print X.shape, Y.shape
f_voc = open('/mnt/data/hog.save', 'wb')
cPickle.dump((X, Y), f_voc, protocol=cPickle.HIGHEST_PROTOCOL)
f_voc.close()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

clf.score(X_test, y_test)





