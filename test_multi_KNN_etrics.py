##KNeighborsClassifier, you can specify a different distance metric using the metric parameter. Here are some common options:Euclidean Distance (default): metric='euclidean' Manhattan Distance: metric='manhattan'Minkowski Distance: metric='minkowski' (generalization of Euclidean and Manhattan) Chebyshev Distance: metric='chebyshev' Example: Testing Different Distance Metrics in KNN You can modify the KNN distance metric in your ensemble method like this:


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

# Define the Naive Bayes classifier
nb = GaussianNB()

# List of distance metrics to test for KNN
distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
best_metric = None
best_accuracy = 0

# Iterate over different distance metrics for KNN
for metric in distance_metrics:
    # Define the KNN classifier with the current distance metric and the best K value (K=5)
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    
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
    
    # Print the accuracy for the current distance metric
    print(f"Distance Metric = {metric}, Accuracy: {accuracy * 100:.2f}%")
    
    # Keep track of the best distance metric
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_metric = metric

# Print the best distance metric and corresponding accuracy
print(f"\nBest Distance Metric: {best_metric}, with Accuracy: {best_accuracy * 100:.2f}%")

