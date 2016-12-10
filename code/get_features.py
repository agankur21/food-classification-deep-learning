import os

import caffe
import numpy as np
import pickle

input_image_folder = "/mnt/data/train"
output_file = '/mnt/data/f7_features.p'
model_file = '/mnt/data/bvlc_caffenet/bvlc_caffenet_iter_2000.caffemodel'
deploy_prototxt = '/home/ubuntu/git-repo/food-classification-deep-learning/caffe_models/bvlc_caffenet/deploy.prototxt'
imagemean_file = '/mnt/data/mean_all.npy'
net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)
layer = 'fc7-food'
if layer not in net.blobs:
    raise TypeError("Invalid layer name: " + layer)
print net.blobs['data'].data.shape
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

net.blobs['data'].reshape(1, 3, 227, 227)
output_dict={}

def get_files_processed(output_file):
    if os.path.isfile(output_file) is False:
        return set([])
    lines = open(output_file, 'r').read().splitlines()
    files = set([line.split('\t')[0] for line in lines])
    return files


def save_features_for_all_files():
    list_img_files = os.listdir(input_image_folder)
    try:
        for img_file in list_img_files:
            save_image_features(img_file, input_image_folder)
    except Exception as e:
        print e.message
    finally:
        pickle.dump(output_dict,open(output_file,'wb'))

def save_image_features(image_file, image_folder):
    complete_input_image_path = os.path.join(image_folder, image_file)
    if os.path.isfile(complete_input_image_path) is False:
        print "Incorrect path : " + complete_input_image_path
        return
    print "Processing file: " + complete_input_image_path
    img = caffe.io.load_image(complete_input_image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    output = net.forward()
    output_dict[image_file] = net.blobs[layer].data[0]
    print "Features for file: %s added to dictionary" %complete_input_image_path

if __name__ == '__main__':
    save_features_for_all_files()