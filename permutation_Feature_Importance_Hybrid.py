from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import time

# Load the data from result.csv
data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')
X = data[:, 0:5]  # Features (0 to 4)
y = data[:, 5]    # Target (5th column)

# Split the data into training and test sets (75% train, 25% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# Define the combined classifier using VotingClassifier with soft voting
combined_classifier = VotingClassifier(estimators=[
    ('GaussianNB', GaussianNB()),
    ('KNN', KNeighborsClassifier(n_neighbors=5))
], voting='soft')

# Fit the combined classifier
combined_classifier.fit(x_train, y_train)

# Calculate accuracy
accuracy = accuracy_score(y_test, combined_classifier.predict(x_test))
print(f"Combined Classifier (Gaussian Naive Bayes and KNN) Accuracy: {accuracy * 100:.2f}%")

# Calculate permutation feature importance
result = permutation_importance(combined_classifier, x_test, y_test, n_repeats=10, random_state=0)

# Get importance values
importance_means = result.importances_mean
importance_std = result.importances_std
feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']  # Adjust to match your actual feature names

# Plot Feature Importance
plt.figure(figsize=(8, 6))
plt.barh(feature_names, importance_means, xerr=importance_std, color='orange')
plt.title('Feature Importance for Hybrid Voting Classifier (KNN + NB)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

