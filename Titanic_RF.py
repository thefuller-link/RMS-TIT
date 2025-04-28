# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 15:22:18 2025

@author: lfull
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('train.csv')

# Drop unwanted columns first
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# List of columns for each transformation
numeric_features = ['Age', 'Fare']
categorical_features = ['Sex', 'Pclass']

# Preprocessing pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Preprocessing pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)])

# Full pipeline: preprocessing + model
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
clf.fit(X_train, y_train)

# Score it
print("Model Accuracy:", clf.score(X_test, y_test))

test_df = pd.read_csv("test.csv")
passenger_ids = test_df["PassengerId"]
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

predictions = clf.predict(test_df)

submission = pd.DataFrame({'PassengerId': passenger_ids,
                           'Survived': predictions})

submission.to_csv('submission_RF_only.csv', index=False)