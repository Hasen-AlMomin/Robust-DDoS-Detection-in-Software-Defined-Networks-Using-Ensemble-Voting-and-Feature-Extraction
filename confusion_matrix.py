from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier  # Added this import
import time



from sklearn.model_selection import cross_val_score

#step1: Load the data in numpy array
data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')
X = data[:, 0:5]
y = data[:, 5]

#step2: Split the data to training & test data. Test-size is 0.25(25%) of data
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)


#step3: select the machine learning algorithm
#svm
#clf = svm.SVC()
#clf = svm.SVC(kernel="linear",C=0.025)
#clf = svm.SVC(kernel="linear")
#clf = svm.SVC(gamma=2, C=1)

#Decision Tree
#clf = tree.DecisionTreeClassifier()
#clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)


#Gaussian Naive Bayes
#clf = GaussianNB()   

#Forests of randomized trees
clf = RandomForestClassifier()

#Tree Classifier
#clf = tree.DecisionTreeClassifier()

#neural network classifier
#clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(6, 5, 10), random_state=1)


#KNN 
#clf = KNeighborsClassifier()


#Naive Bayes
#clf = GaussianNB()

# Define the combined classifier using VotingClassifier with soft voting
#clf = VotingClassifier(estimators=[
#    ('GaussianNB', GaussianNB()),
#    ('KNN', KNeighborsClassifier())
#], voting='soft')


#step4: Train the ML Algo with training data
clf.fit(x_train, y_train)


#step5: Pass the test data for classify or predict
classifier_predictions = clf.predict(x_test)


#step6. Calculate the confusion matrix

print("Actual ", y_test)
print("predictions ", classifier_predictions)


tn, fp, fn, tp = confusion_matrix(y_test, classifier_predictions).ravel()
print("true negative", tn)
print("false positive", fp)
print("false negative", fn)	
print("true positive",  tp)  



print(confusion_matrix(y_test, classifier_predictions))


precision = tp/(tp + fn)

#recall = (tp+tn)/(tp+tn+fn+fp) 

print ("Precision Ratio = ", precision)

#print("Recall(Sesetivity) = " , recall)
'''
#step6. Calculate the accuracy from the the prediction result.
print("Accuracy is ", accuracy_score(y_test, classifier_predictions)*100)


#step7. calculate cross validation score
scores = cross_val_score(clf, x_train, y_train, cv=5)
print("cross-validation score",scores.mean())

'''
