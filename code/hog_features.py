import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn import svm
from sklearn.cross_validation import train_test_split
import cPickle
from scipy import misc
import os

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
        image = np.atleast_2d(im)

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




def get_files_processed(output_file):
    if os.path.isfile(output_file) is False:
        return set([])
    lines = open(output_file, 'r').read().splitlines()
    files = set([line.split(',')[0] for line in lines])
    return files


def calculate_save_hog_features(image_dir,output_file_path):
    list_img_files = os.listdir(image_dir)
    processed_files = get_files_processed(output_file_path)
    out_file = open(output_file_path, mode='a')
    try:
        count_files=0
        for img_file in list_img_files:
            if img_file in processed_files:
                continue
            complete_img_path = os.path.join(image_dir,img_file)
            img = misc.imread(complete_img_path)
            features = hog_feature(img)
            feature_str = ",".join((features[np.newaxis,:]).astype('str')[0])
            out_file.write(img_file + ',' + feature_str + '\n')
            print "File Processed : " + os.path.join(image_dir, img_file)
            count_files += 1
            if count_files % 100 == 0:
                print "Number of Files Processed : " + str(count_files)
    except Exception as e:
        print e.message
    finally:
        out_file.close()


if __name__ == '__main__':
    calculate_save_hog_features("/mnt/data/train", '/mnt/data/hog_features_train.txt')
    calculate_save_hog_features("/mnt/data/test", '/mnt/data/hog_features_test.txt')






