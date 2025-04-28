# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 17:10:42 2025

@author: lfull
"""

import pandas as pd
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess training data
df = pd.read_csv("train.csv")
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

X = df.drop('Survived', axis=1)
y = df['Survived']

# Define preprocessing
numeric_features = ['Age', 'Fare']
categorical_features = ['Sex', 'Pclass']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)])

# === Cross-validation loop ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_scores = []
nn_scores = []
blend_scores = []
y_true_all = [] #for classification/cm reports
y_pred_all = [] #for classification/cm reports

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n Fold {fold + 1}")
    
    X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Preprocess each fold independently
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    # --- Random Forest ---
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_val)
    rf_probs = rf_model.predict_proba(X_val)
    rf_acc = accuracy_score(y_val, rf_preds)
    rf_scores.append(rf_acc)
    print(f"RF Accuracy: {rf_acc:.4f}")

    # --- Neural Network ---
    y_train_cat = to_categorical(y_train)
    nn_model = Sequential()
    nn_model.add(Input(shape=(X_train.shape[1],)))
    nn_model.add(Dense(256, activation="relu"))
    nn_model.add(Dropout(0.2))
    nn_model.add(Dense(128, activation="relu"))
    nn_model.add(Dropout(0.2))
    nn_model.add(Dense(64, activation="relu"))
    nn_model.add(Dense(2, activation="sigmoid"))
    nn_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    nn_model.fit(X_train, y_train_cat, epochs=50, batch_size=8, verbose=0)
    
    nn_probs = nn_model.predict(X_val)
    nn_preds = np.argmax(nn_probs, axis=1)
    nn_acc = accuracy_score(y_val, nn_preds)
    nn_scores.append(nn_acc)
    print(f"NN Accuracy: {nn_acc:.4f}")

    # --- Blended Predictions ---
    blended_probs = (rf_probs + nn_probs) / 2
    blended_preds = np.argmax(blended_probs, axis=1)
    y_true_all.extend(y_val) # for classifation/CM reports
    y_pred_all.extend(blended_preds) #for classification/cm reports
    blended_acc = accuracy_score(y_val, blended_preds)
    blend_scores.append(blended_acc)
    print(f"Blended Accuracy: {blended_acc:.4f}")
    
    

# === Final Summary ===
print("\n Cross-Validation Summary:")
print(f"Avg RF Accuracy:      {np.mean(rf_scores):.4f}")
print(f"Avg NN Accuracy:      {np.mean(nn_scores):.4f}")
print(f"Avg Blended Accuracy: {np.mean(blend_scores):.4f}")

print("\n=== Overall Blended Model Evaluation Across All Folds ===")
blended_cm = confusion_matrix(y_true_all, y_pred_all)
print("Confusion Matrix:")
print(blended_cm)
print("Classification Report:")
print(classification_report(y_true_all, y_pred_all, digits=4))

# Optional: Plot the confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(blended_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title(f'Confusion Matrix - Blended Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()


# === Final Training on Full Data for Submission ===
X_full = preprocessor.fit_transform(X)
y_full_cat = to_categorical(y)

# Final RF
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_full, y)

# Final NN
nn_model = Sequential()
nn_model.add(Input(shape=(X_full.shape[1],)))
nn_model.add(Dense(128, activation="relu"))
nn_model.add(Dropout(0.3))
nn_model.add(Dense(64, activation="relu"))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(64, activation="relu"))
nn_model.add(Dense(2, activation="softmax"))
nn_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
nn_model.fit(X_full, y_full_cat, epochs=100, batch_size=5, verbose=0)

# Save model
nn_model.save("titanic.keras")
load_model("titanic.keras")

# === Predict on Test Set ===
test_df = pd.read_csv("test.csv")
passenger_ids = test_df["PassengerId"]
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
X_test_processed = preprocessor.transform(test_df)

# Predict and blend
rf_probs_test = rf_model.predict_proba(X_test_processed)
nn_probs_test = nn_model.predict(X_test_processed)
blended_probs_test = (rf_probs_test + nn_probs_test) / 2
blended_preds_test = np.argmax(blended_probs_test, axis=1)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': blended_preds_test
})
submission.to_csv('submission_nn_rf.csv', index=False)
