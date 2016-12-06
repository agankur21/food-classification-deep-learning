import sys
import numpy as np
import matplotlib.pyplot as plt

sys.insert('/path/to/caffe/python')
import caffe


caffe.set_device(0)
caffe.set_mode_gpu()