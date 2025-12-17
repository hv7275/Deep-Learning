import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import Perceptron

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings(
  'ignore'
)

df = sns.load_dataset('iris')
print(df)
print()

print(df.species.value_counts())
print()

plt.figure(figsize=(16, 12))
sns.pairplot(df, hue='species')
plt.show()

X = df.drop(
  ['species'],
  axis=1
)

y = df['species']

label_enc = LabelEncoder()
y_int = label_enc.fit_transform(y)
print(y_int)
print()

scaler = StandardScaler()
X_scaled = pd.DataFrame(
  scaler.fit_transform(X),
  columns=X.columns,
  index=df.index
)
print(X_scaled)
print()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_int,
    test_size=0.2,
    random_state=42,
    stratify=y_int
)
print(X_train.shape, X_test.shape)


percep = Perceptron(
  max_iter=1000,
  random_state=42
)

percep.fit(X_train, y_train)
percep_pred = percep.predict(X_test)

acc = accuracy_score(y_test, percep_pred)
print(f"Accuracy: {acc:.3f}")

print(classification_report(
    y_test,
    percep_pred,
    target_names=label_enc.classes_
))

cm = confusion_matrix(y_test, percep_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_enc.classes_,
    yticklabels=label_enc.classes_
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
