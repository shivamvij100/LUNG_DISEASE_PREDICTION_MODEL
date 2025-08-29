# First, save the provided CSV data to a file named 'lung.csv' in your working directory.
# The content is the CSV inside the <DOCUMENT> tag.

# Script to train and save the model to 'lung_disease_pred.pkl'
# Run this script once to create the .pkl file.

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # Using RandomForest as an example; you can replace with other classifiers like LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('lung.csv')

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Optional: Split for evaluation (but since the original doesn't, we train on full data)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)  # Or fit(X_train, y_train) if splitting

# Optional: Evaluate
# pred = model.predict(X_test)
# print(accuracy_score(y_test, pred))

# Save the model
with open('lung_disease_pred.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved to 'lung_disease_pred.pkl'")