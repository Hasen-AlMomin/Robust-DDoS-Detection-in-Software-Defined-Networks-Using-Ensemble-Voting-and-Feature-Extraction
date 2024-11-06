from __future__ import division
import numpy
import os
from sklearn.linear_model import LogisticRegression
from collections import deque
#import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_decision_regions
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier  # Added this import
import time


class MachineLearningAlgo:
    def __init__(self):
        """
        train the model from generated training data in generate-data folder
        """
        self.data = numpy.loadtxt(open('result.csv', 'rb'), delimiter=',', dtype=float)
        #self.clf = svm.SVC(kernel="linear")
        #self.clf = svm.SVC()
        #self.clf = svm.SVC(gamma=2, C=1)

        #Decision Tree
        #self.clf = tree.DecisionTreeClassifier()
        #self.clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)

        #Gaussian Naive Bayes
        #self.clf = GaussianNB()   
        
        
        #k-nearest Near
        self.clf = KNeighborsClassifier()


        #Forests of randomized trees
        #self.clf = RandomForestClassifier(n_estimators=10)

        #neural network classifier
        #self.clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

	# Define the combined classifier using VotingClassifier with soft voting
	#self.clf = VotingClassifier(estimators=[
	#('GaussianNB', GaussianNB()),
	#('KNN', KNeighborsClassifier())
	#], voting='soft') 

        #self.clf = VotingClassifier(estimators=[('ANN', MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(6,5, 10), random_state=1)), ('KNN', KNeighborsClassifier())], voting='soft')

        #self.clf = VotingClassifier(estimators=[('GaussianNB', GaussianNB()), ('KNN', KNeighborsClassifier())], voting='soft')


        # train the model - y values are locationed in last (index 3) column
        self.clf.fit(self.data[:, 0:5], self.data[:, 5])


    def classify(self, data):
        fparams = numpy.zeros((1, 5))
        fparams[:,0] = data[0]
        fparams[:,1] = data[1]
        fparams[:,2] = data[2]
        fparams[:,3] = data[3]
        fparams[:,4] = data[4]        
        prediction = self.clf.predict(fparams)
        #print("SVM input data", data , "prediction result ", prediction)
        return prediction
