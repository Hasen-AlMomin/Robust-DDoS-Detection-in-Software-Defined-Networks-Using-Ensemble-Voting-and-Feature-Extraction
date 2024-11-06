#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#https://go2analytics.wordpress.com/tag/roc/
from itertools import cycle
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier  # Added this import

#step1: Load the data in numpy array
data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')


#step2: Split the data to training & test data. Test-size is 0.25(25%) of data
X = data[:, 0:5]
y = data[:, 5]
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.5)#clf = svm.SVC()

n_samples, n_features = X.shape


'''
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
n_samples, n_features = X.shape
'''

#step3: select the machine learning algorithm

#svm
#clf = svm.SVC(kernel='linear',probability=True)
#clf = svm.SVC(kernel="linear",C=0.025)
#clf = svm.SVC(kernel="linear")
#sclf = svm.SVC(probability=True)
#clf = svm.SVC(probability=True)
#Decision Tree
#dclf = tree.DecisionTreeClassifier()
#clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
dclf = GaussianNB()


#Gaussian Naive Bayes
#gclf = GaussianNB()   
gclf = KNeighborsClassifier()

#Forests of randomized trees
#sclf = RandomForestClassifier(n_estimators=10)

# Define the combined classifier using VotingClassifier with soft voting
sclf = VotingClassifier(estimators=[
    ('GaussainNB', GaussianNB()),
    ('KNN', KNeighborsClassifier(n_neighbors=5))
], voting='soft')

#Extra Tree Classifier
#clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)

#neural network classifier
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)


#step4: Train the ML Algo with training data
sprobas_ = sclf.fit(x_train, y_train).predict_proba(x_test)
dprobas_ = dclf.fit(x_train, y_train).predict_proba(x_test)
gprobas_ = gclf.fit(x_train, y_train).predict_proba(x_test)

# Compute ROC curve and area under the curve
sfpr, stpr, sthresholds = roc_curve(y_test, sprobas_[:, 1])
sroc_auc = auc(sfpr, stpr)
print("Enesemble Voting Area under the ROC curve : %f" % sroc_auc)

# Compute ROC curve and area under the curve
dfpr, dtpr, dthresholds = roc_curve(y_test, dprobas_[:, 1])
droc_auc = auc(dfpr, dtpr)
print("Decision Tree Area under the ROC curve : %f" % droc_auc)


# Compute ROC curve and area under the curve
gfpr, gtpr, gthresholds = roc_curve(y_test, gprobas_[:, 1])
groc_auc = auc(gfpr, gtpr)
print("K-Nearest Neighbors under the ROC curve : %f" % groc_auc)


plt.figure()
lw = 2
#plt.plot(fpr[2], tpr[2], color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot(sfpr, stpr, color='darkorange',lw=lw, label='Ensemble Voting (KNN+NB) ROC curve (area = %0.2f)' % sroc_auc)
plt.plot(dfpr, dtpr, color='darkblue',lw=lw, label='Decision Tree ROC curve (area = %0.2f)' % droc_auc)
plt.plot(gfpr, gtpr, color='darkgreen',lw=lw, label='K-Nearest Neighbors ROC curve (area = %0.2f)' % droc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("roc_enesemble_voting_best_3.png")
