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

# Define the KNN and Naive Bayes classifiers
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()

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

# Print the accuracy score
print(f"Accuracy of KNN + Naive Bayes Voting Ensemble (Soft Voting): {accuracy * 100:.2f}%")

