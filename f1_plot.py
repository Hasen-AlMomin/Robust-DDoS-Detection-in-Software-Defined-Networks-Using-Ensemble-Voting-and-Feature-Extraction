from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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

# Make predictions
predictions = combined_classifier.predict(x_test)

# Calculate the F1 score
f1 = f1_score(y_test, predictions, average='binary')  # Adjust 'average' based on your dataset (e.g., 'micro', 'macro', 'weighted')

# Print the F1 score
print(f"F1 Score for Combined Classifier (Gaussian Naive Bayes and KNN): {f1 * 100:.2f}%")

# Plot F1 score (single bar for the ensemble classifier)
plt.figure(figsize=(6, 4))
plt.bar(['Combined Classifier (KNN + NB)'], [f1 * 100], color='orange')
plt.title('F1 Score for Hybrid Voting Classifier (KNN + NB)')
plt.ylabel('F1 Score (%)')
plt.ylim(0, 100)

# Display the F1 score on the bar
for i, v in enumerate([f1 * 100]):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')

# Show the plot
plt.show()

