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

# Parameter combinations to test for KNN (with Manhattan distance)
knn_params = {
    'n_neighbors': [3, 5, 7],            # Number of neighbors to test
    'weights': ['uniform', 'distance'],   # Weighting scheme to test
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],  # Algorithms to test
    'leaf_size': [20, 30, 40],            # Leaf size to test for BallTree or KDTree
}

best_params = None
best_accuracy = 0

# Iterate over the combinations of KNN parameters
for n_neighbors in knn_params['n_neighbors']:
    for weight in knn_params['weights']:
        for algorithm in knn_params['algorithm']:
            for leaf_size in knn_params['leaf_size']:
                # Define the KNN classifier with the current parameter combination
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='manhattan',
                                           weights=weight, algorithm=algorithm, leaf_size=leaf_size)
                
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
                
                # Print the accuracy for the current parameter combination
                print(f"n_neighbors={n_neighbors}, weights={weight}, algorithm={algorithm}, leaf_size={leaf_size}, Accuracy: {accuracy * 100:.2f}%")
                
                # Keep track of the best parameter combination
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (n_neighbors, weight, algorithm, leaf_size)

# Print the best parameters and corresponding accuracy
print(f"\nBest Parameters: n_neighbors={best_params[0]}, weights={best_params[1]}, algorithm={best_params[2]}, leaf_size={best_params[3]}, with Accuracy: {best_accuracy * 100:.2f}%")

