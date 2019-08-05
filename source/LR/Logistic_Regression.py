# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:02:18 2018

@author: Manjunath Dharshan
"""
import time
import load_dataset as ld
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


class LogisticRegression(object):
    def __init__(self):
        self.number_of_iterations = 100
        self.number_of_labels = 10
        self.training_size = 60000
        self.testing_size = 10000
        self.alpha = 0.033

    def load_train_test(self):
        Y_train, X_train = ld.read()
        self.Y_test, self.X_test = ld.read('testing')
        X_train = np.reshape(X_train,(self.training_size,28*28))/255
        self.X_test =  np.reshape(self.X_test,(self.testing_size,28*28))/255
        Y_train = self.one_hot_representation(Y_train)
        #Y_test = self.one_hot_representation(Y_test)
        return (X_train, Y_train, self.X_test, self.Y_test)

    def one_hot_representation(self, array):
        array_shape = len(array)
        sparse_one_hot = sparse.csr_matrix((np.ones(array_shape), (np.arange(array_shape), array)),
                                           shape=(array_shape, self.number_of_labels))
        return sparse_one_hot.todense()

    def softmax(self, z):
        z -= np.max(z)
        sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
        return sm

    def gradient_ascent(self, X_train, Y_train, weights):
        iters = []
        accuracies=[]
        for i in range(self.number_of_iterations):
            print("iteration:::", i)
            h = np.einsum('ij,jk->ik',X_train,weights)
            h_train = self.softmax(h)

            # predicted_labels = self.predict(weights, X_train)
            # accuracy = self.getAccuracy(Y_train, predicted_labels) * 100
            # print("train acc",accuracy)

            gradients = np.einsum("ij,jk->ik",X_train.T,Y_train -h_train)
            #gradient = np.dot(X_train.T, h - Y_train) / training_size
            h_test_weight = np.einsum('ij,jk->ik', self.X_test, weights)
            #h_test = self.softmax(h_test_weight)
            predicted_labels = self.predict(weights, self.X_test)
            accuracy = self.getAccuracy(self.Y_test, predicted_labels) * 100
            print("accuracy:::",accuracy)
            accuracies.append(accuracy)
            gradient = gradients / self.training_size
            weights = weights + (gradient * self.alpha)
            iters.append(i)
        plt.xlabel("Number of iterations")
        plt.ylabel("Training Accuracy %")
        plt.plot(iters, accuracies, marker='o')
        plt.show()
        return weights

    def train(self, X_train, Y_train):
        X_train_copy = np.copy(X_train)
        Y_train_copy = np.copy(Y_train)
        weights = np.random.random((X_train.shape[1], 10)) * (0.001)
        weights = self.gradient_ascent(X_train, Y_train, weights)
        return weights

    def getAccuracy(self, Y_test, predicted_labels):
        #prob, prede = self.getProbsAndPreds(someX,someY.T)
        argmaxes = np.argmax(predicted_labels, axis = 1)
        print("argmax:::",argmaxes)
        accuracy = np.sum(argmaxes == Y_test) / (float(len(Y_test)))
        return accuracy

    def predict(self, all_weights, X_test):
        predicted_labels = np.dot(all_weights.T, X_test.T)
        print(predicted_labels.shape)
        predicted_labels = self.softmax(predicted_labels)
        print(predicted_labels.shape)
        return predicted_labels.T

    def main(self):
        X_train, Y_train, X_test, Y_test = self.load_train_test()
        start_time = time.clock()
        all_weights = self.train(X_train, Y_train)
        # print("Training Time: %.2f seconds" % (time.clock() - start_time))
        # print("Weights Learned!")
        # print("Classifying Test Images ...")
        # start_time = time.clock()
        # #print(all_weights.shape)
        # predicted_labels = self.predict(all_weights, X_test)
        # print("Prediction Time: %.2f seconds" % (time.clock() - start_time))
        # print("Test Images Classified!")
        # #print("predicted_label",predicted_labels.shape)
        # #print("y_test",Y_test.shape)
        # accuracy = self.getAccuracy(Y_test, predicted_labels) * 100
        # print(accuracy)


if __name__ == '__main__':
    lr = LogisticRegression()
    lr.main()