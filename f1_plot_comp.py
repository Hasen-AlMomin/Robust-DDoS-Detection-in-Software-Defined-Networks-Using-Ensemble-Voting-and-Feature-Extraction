from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the data from result.csv
data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')
X = data[:, 0:5]  # Features (0 to 4)
y = data[:, 5]    # Target (5th column)

# Split the data into training and test sets (75% train, 25% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# Define individual classifiers
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Ensemble (KNN + NB)": VotingClassifier(estimators=[
        ('GaussianNB', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_neighbors=5))
    ], voting='soft')
}

# Fit each classifier and calculate F1 score
f1_scores = []
for name, clf in classifiers.items():
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    f1 = f1_score(y_test, predictions, average='binary')  # Change 'binary' if it's multi-class
    f1_scores.append((name, f1 * 100))

# Sort by F1 score for better visualization
f1_scores.sort(key=lambda x: x[1], reverse=True)

# Plot the F1 scores
classifier_names = [x[0] for x in f1_scores]
f1_values = [x[1] for x in f1_scores]

plt.figure(figsize=(8, 5))
plt.barh(classifier_names, f1_values, color='skyblue')
plt.xlabel('F1 Score (%)')
plt.title('F1 Score Comparison for Different Classifiers')

# Display the F1 score on the bars
for index, value in enumerate(f1_values):
    plt.text(value + 1, index, f"{value:.2f}%", va='center')

plt.xlim(0, 100)  # Limit the x-axis from 0 to 100
plt.tight_layout()
plt.show()

