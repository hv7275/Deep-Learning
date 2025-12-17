import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')


# Load Dataset
df = sns.load_dataset('iris')

# Features and target
X = df.drop('species', axis=1)
y = df['species']



# Encode Labels (integer encoding)
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)  # 0,1,2



# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Train / Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_int,
    test_size=0.2,
    random_state=42,
    stratify=y_int
)



# Build ANN Model
ann_model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])



# Compile Model
# Use sparse_categorical_crossentropy for integer labels

ann_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

ann_model.summary()



# Train Model
history = ann_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    verbose=1
)


# Evaluate Model
y_pred_proba = ann_model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()



# Plot Training History
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training History')
plt.legend()
plt.tight_layout()
plt.show()
