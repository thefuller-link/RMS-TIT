import os
os.environ["KERAS_BACKEND"] = "torch"

import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# === Load and preprocess training data ===
df = pd.read_csv("train.csv")
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Visual: Survival Rate by Sex and Class
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex") #, ci=None)
plt.title("Survival Rate by Passenger Class and Sex")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.ylim(0, 1)
plt.legend(title="Sex")
plt.tight_layout()
plt.show()


X = df.drop("Survived", axis=1)
y = df["Survived"]




numeric_features = ['Age', 'Fare']
categorical_features = ['Sex', 'Pclass']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))  # <-- fixed here
])

# "sparse_output=False" for Lincoln's Spyder version
# "sparse=False" for BB's Spyder version

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# === Cross-validation ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nn_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}")
    X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    y_train_cat = to_categorical(y_train)
    
    # === Define Neural Network ===
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="sigmoid"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train_cat, epochs=100, batch_size=10, verbose=0)

    nn_preds = np.argmax(model.predict(X_val), axis=1)
    acc = accuracy_score(y_val, nn_preds)
    nn_scores.append(acc)
    print(f"Validation Accuracy: {acc:.4f}")

print(f"\nAverage NN Accuracy: {np.mean(nn_scores):.4f}")

# === Final Model Training on Full Data ===
X_full = preprocessor.fit_transform(X)
y_full_cat = to_categorical(y)

final_model = Sequential()
final_model.add(Input(shape=(X_full.shape[1],)))
final_model.add(Dense(128, activation="relu"))
final_model.add(Dropout(0.2))
final_model.add(Dense(64, activation="relu"))
final_model.add(Dropout(0.2))
final_model.add(Dense(2, activation="sigmoid"))
final_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
final_model.fit(X_full, y_full_cat, epochs=100, batch_size=10, verbose=0)

# Save model
final_model.save("titanic_nn_only.keras")

# === Make predictions on test set ===
test_df = pd.read_csv("test.csv")
passenger_ids = test_df["PassengerId"]
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
X_test = preprocessor.transform(test_df)

test_preds = np.argmax(final_model.predict(X_test), axis=1)

# Submission file
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": test_preds
})
submission.to_csv("submission_nn_only.csv", index=False)
