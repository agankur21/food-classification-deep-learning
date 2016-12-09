import sys
import caffe
import numpy as np

input_image_file ='/mnt/data/sushi_819.jpg'
output_file= '/mnt/data/test.out'
model_file = '/home/ubuntu/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
deploy_prototxt='/home/ubuntu/git-repo/food-classification-deep-learning/caffe_models/bvlc_caffenet/deploy.prototxt'

net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)
layer = 'fc7'
if layer not in net.blobs:
	raise TypeError("Invalid layer name: " + layer)
print net.blobs['data'].data.shape
imagemean_file = '/mnt/data/mean_all.binaryproto'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)
net.blobs['data'].reshape(1,3,227,227)
img = caffe.io.load_image(input_image_file)
net.blobs['data'].data[...] = transformer.preprocess('data', img)
output = net.forward()
with open(output_file, 'w') as f:
    np.savetxt(f, net.blobs[layer].data[0], fmt='%.4f', delimiter=',')


