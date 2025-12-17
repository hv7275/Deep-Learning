import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# Create Dataset
df = pd.DataFrame({
    "soil_moisture": [0.10, 0.15, 0.20, 0.25, 0.40, 0.60, 0.35, 0.18,
                      0.45, 0.05, 0.80, 0.27, 0.55, 0.70, 0.12, 0.30],
    
    "temperature_c": [34, 30, 26, 22, 28, 30, 19, 22,
                      35, 24, 33, 33, 21, 25, 20, 29],
    
    "sunlight_hours": [9, 8, 7, 4, 8, 10, 3, 10,
                       12, 5, 9, 11, 2, 6, 1, 9],
    
    "needs_water": [1, 1, 1, 0, 0, 0, 0, 1,
                    0, 1, 0, 1, 0, 0, 1, 1]
})



# Split Features and Target
X = df[['soil_moisture', 'temperature_c', 'sunlight_hours']]
y = df['needs_water']



# Feature Scaling (0–1 range)
scaler = MinMaxScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)


# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)



# Build Neural Network Model
model = keras.Sequential([
    layers.Input(shape=(X_scaled.shape[1],)),  # number of features
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')      # binary output
])



# Compile Model
optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# Train Model
history = model.fit(
    X_train.values,
    y_train.values,
    validation_data=(X_test.values, y_test.values),
    epochs=100,
    batch_size=4,
    verbose=1
)


# Plot Training History
epochs = range(len(history.history['accuracy']))

plt.figure(figsize=(8, 5))
sns.lineplot(x=epochs, y=history.history['accuracy'], label='Training Accuracy')
sns.lineplot(x=epochs, y=history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Epochs")
plt.legend()
plt.show()


# Make Predictions on Test Set
# Predict probabilities
y_pred_proba = model.predict(X_test)

# Convert probabilities to class labels (0 or 1)
y_pred = (y_pred_proba >= 0.5).astype(int).ravel()



# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
