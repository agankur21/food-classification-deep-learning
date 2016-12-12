import os

import caffe
import numpy as np
import cPickle

model_file = '/mnt/data/Training_Snapshot/snapshot5/bvlc_caffenet_iter_8000.caffemodel'
deploy_prototxt = '/mnt/data/Training_Snapshot/snapshot5/deploy.prototxt'
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
num_features_out=net.blobs[layer].data.shape[1]

def get_label_dict(label_file):
    lines= open(label_file).read().splitlines()
    dict_labels={ x.split(' ')[0].strip() : x.split(' ')[1].strip() for x in lines}
    return dict_labels

def get_files_processed(output_file):
    if os.path.isfile(output_file) is False:
        return set([])
    lines = open(output_file, 'r').read().splitlines()
    files = set([line.split(',')[0] for line in lines])
    return files


def save_features_for_all_files(input_image_folder,label_file,output_file):
    labels_dict=get_label_dict(label_file)
    list_img_files = os.listdir(input_image_folder)
    out_file = open(output_file, mode='wb')
    X =np.zeros((len(list_img_files),num_features_out))
    y=[]
    image_files=[]
    try:
        count_files= 0
        for img_file in list_img_files:
            if img_file not in labels_dict:
                continue
            image_features= get_image_features(img_file, input_image_folder)
            np.copyto(X[count_files,:],image_features)
            y.append(labels_dict[img_file])
            image_files.append(img_file)
            print "File Processed : "+ os.path.join(input_image_folder,img_file)
            count_files += 1
            if count_files %100 == 0:
                print "Number of Files Processed : " + str(count_files)

    except Exception as e:
        print e.message
    finally:
        cPickle.dump((X,np.array(y),np.array(image_files)), out_file, protocol=cPickle.HIGHEST_PROTOCOL)
        out_file.close()


def get_top_arguments(numpy_array,num_arguments=5):
    sorted_arguments= np.argsort(numpy_array)[-num_arguments:][::-1]
    return sorted_arguments


def get_image_features(image_file, image_folder):
    complete_input_image_path = os.path.join(image_folder, image_file)
    if os.path.isfile(complete_input_image_path) is False:
        print "Incorrect path : " + complete_input_image_path
        return
    img = caffe.io.load_image(complete_input_image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    output = net.forward()
    return net.blobs[layer].data[0]


if __name__ == '__main__':
    label_file = '/mnt/data/train_data.txt'
    save_features_for_all_files("/mnt/data/train",label_file,'/mnt/data/f7_features_train.p')
    label_file = '/mnt/data/test_data.txt'
    save_features_for_all_files("/mnt/data/test",label_file,'/mnt/data/f7_features_test.p')
