import os

import caffe
import numpy as np

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
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255.0)

net.blobs['data'].reshape(1, 3, 227, 227)


def get_files_processed(output_file):
    if os.path.isfile(output_file) is False:
        return set([])
    lines = open(output_file, 'r').read().splitlines()
    files = set([line.split(',')[0] for line in lines])
    return files


def save_features_for_all_files(input_image_folder,output_file):
    list_img_files = os.listdir(input_image_folder)
    processed_files = get_files_processed(output_file)
    out_file = open(output_file, mode='a')
    try:
        count_files= 0
        total_files=len(list_img_files)
        for img_file in list_img_files:
            if img_file in processed_files:
                continue
            save_image_features(img_file, input_image_folder, out_file)
            print "File Processed : "+ os.path.join(input_image_folder,img_file)
            count_files += 1
            if count_files %100 == 0:
                print "Number of Files Processed : " + count_files

    except Exception as e:
        print e.message
    finally:
        out_file.close()


def save_image_features(image_file, image_folder, out_file):
    complete_input_image_path = os.path.join(image_folder, image_file)
    if os.path.isfile(complete_input_image_path) is False:
        print "Incorrect path : " + complete_input_image_path
        return
    img = caffe.io.load_image(complete_input_image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    output = net.forward()
    feature = ",".join((net.blobs[layer].data[0][np.newaxis, :]).astype('str'))
    out_file.write(image_file + ',' + feature + '\n')


if __name__ == '__main__':
    save_features_for_all_files("/mnt/data/train",'/mnt/data/f7_features_train.txt')
    save_features_for_all_files("/mnt/data/test",'/mnt/data/f7_features_test.txt')