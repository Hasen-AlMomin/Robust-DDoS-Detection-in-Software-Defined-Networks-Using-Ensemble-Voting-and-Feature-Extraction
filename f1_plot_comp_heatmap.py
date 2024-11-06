import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

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

# Initialize lists to store metrics
f1_scores = []
precision_scores = []
recall_scores = []

# Fit each classifier and calculate F1, precision, and recall scores
for name, clf in classifiers.items():
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    
    # Calculate F1, precision, and recall
    f1 = f1_score(y_test, predictions, average='binary')  # Adjust 'binary' for binary classification
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    
    # Append the scores
    f1_scores.append(f1 * 100)
    precision_scores.append(precision * 100)
    recall_scores.append(recall * 100)

# Create a DataFrame for the metrics to plot as a heatmap
metrics_df = pd.DataFrame({
    "Classifier": list(classifiers.keys()),
    "F1 Score": f1_scores,
    "Precision": precision_scores,
    "Recall": recall_scores
})

# Set 'Classifier' as the index for easier plotting
metrics_df.set_index('Classifier', inplace=True)

# Create the heatmap using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(metrics_df, annot=True, cmap='Blues', fmt='.2f', linewidths=.5)

# Add title
plt.title('Heatmap of F1 Scores, Precision, and Recall for Multiple Classifiers')

# Show the plot
plt.tight_layout()
plt.show()

