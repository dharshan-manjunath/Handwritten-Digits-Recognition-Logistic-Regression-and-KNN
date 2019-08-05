# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:02:18 2018

@author: Manjunath Dharshan
"""
import load_dataset as ld
import numpy as np
import time
import matplotlib.pyplot as mp
from scipy.spatial import distance
from scipy.stats import mode

training_size = 60000
testing_size = 10000
class K_N_N(object):
    def __init__(self):
        self.number_of_labels = 10
        self.training_size = 60000
        self.testing_size = 10000

    def load_train_test(self):
        Y_train, X_train = ld.read('training');
        Y_test, X_test = ld.read('testing');
        X_train = np.reshape(X_train,(training_size,28*28))
        X_test =  np.reshape(X_test,(testing_size,28*28))
        return (X_train, Y_train, X_test, Y_test)

    def euclideanDistance(self,X_test, X_train):
        dist = distance.cdist(X_test, X_train, 'euclidean')
        return dist


    def main(self):
        X_train, Y_train, X_test, Y_test = self.load_train_test()
        K = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
        max_k = max(K)
        indices = []
        accuracies = []
        start_time = time.time()
        value  = lr.euclideanDistance(X_test,X_train)

        # find the indices with minimum values
        for i in value:
            indices.append(i.argsort()[:max_k])
        predicted_labels = Y_train[indices]
        for k in K:
            #print("predicted_labels:::",predicted_labels,"shape:::",predicted_labels.shape)
            neighbors = predicted_labels[:,:k]
            #print("neighbors:::", neighbors,"shape::",neighbors.shape)
            predictions = np.squeeze(mode(neighbors,axis = 1)[0])
            #print("predictions:::",predictions,"shape:::",predictions.shape)
            accuracy = np.sum(predictions == Y_test) / len(Y_test)
            #print("accuracy:::",accuracy)
            accuracies.append(accuracy)
        end_time = time.time()
        print("Time taken for execution is {}".format(end_time - start_time))

        # Plotting The Accuracy Curve
        mp.figure(1)
        mp.xlabel("Number of Nearest Neighbors K")
        mp.ylabel("Accuracy %")
        mp.plot(K, accuracies, marker='o', )
        mp.show()
if __name__ == "__main__":
    lr = K_N_N()
    lr.main()