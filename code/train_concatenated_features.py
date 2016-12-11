import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.getcwd(),"../utils"))
from neural_net import *


def train_and_plot(X_train,y_train,X_test,y_test,num_iters=1000,learning_rate=1e-4,reg=0.5):
    input_size=X_train.shape[1]
    hidden_size=input_size
    num_classes=101
    net= TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)
    stats = net.train(X_train, y_train, X_test, y_test,
            num_iters=num_iters, batch_size=256,
            learning_rate=learning_rate, learning_rate_decay=0.95,
            reg=reg, verbose=True)
    # Plot the loss function and train / validation accuracies
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.show()





