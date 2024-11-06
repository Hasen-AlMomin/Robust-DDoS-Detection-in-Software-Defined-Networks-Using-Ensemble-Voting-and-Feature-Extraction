##select the best K for KNN but still stick with the train-test split method (instead of cross-validation), you can perform the following steps:Train the model on the training set using different values of K.Evaluate the model on the test set for each value of K. Select the value of K that gives the highest accuracy on the test set. and best Var-smothing value
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data from result.csv
data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')
X = data[:, 0:5]  # Features (0 to 4)
y = data[:, 5]    # Target (5th column)

# Split the data into training and test sets (75% train, 25% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# Define the KNN classifier with the best K found earlier (K = 5)
knn = KNeighborsClassifier(n_neighbors=5)

# List of `var_smoothing` values to test for Naive Bayes
var_smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
best_var_smoothing = None
best_accuracy = 0

# Iterate over different values of `var_smoothing`
for var_smoothing in var_smoothing_values:
    # Define the Naive Bayes classifier with the current `var_smoothing`
    nb = GaussianNB(var_smoothing=var_smoothing)
    
    # Create the VotingClassifier with soft voting
    ensemble_classifier = VotingClassifier(estimators=[
        ('GaussianNB', nb),
        ('KNN', knn)
    ], voting='soft')
    
    # Fit the model to the training data
    ensemble_classifier.fit(x_train, y_train)
    
    # Make predictions on the test data
    predictions = ensemble_classifier.predict(x_test)
    
    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, predictions)
    
    # Print the accuracy for the current value of `var_smoothing`
    print(f"var_smoothing = {var_smoothing}, Accuracy: {accuracy * 100:.2f}%")
    
    # Keep track of the best `var_smoothing`
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_var_smoothing = var_smoothing

# Print the best `var_smoothing` value and corresponding accuracy
print(f"\nBest var_smoothing: {best_var_smoothing}, with Accuracy: {best_accuracy * 100:.2f}%")

