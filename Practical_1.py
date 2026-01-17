import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, MaxPool1D, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
train = pd.read_csv('/kaggle/input/heartbeat/mitbih_train.csv', header=None)
test  = pd.read_csv('/kaggle/input/heartbeat/mitbih_test.csv', header=None)

# Separate features & labels
X_train_raw = train.iloc[:, :-1].values
y_train_raw = train.iloc[:, -1].astype(int).values
X_test      = test.iloc[:, :-1].values
y_test      = test.iloc[:, -1].astype(int).values

# Manual class balancing (oversampling) 
rng = np.random.default_rng(42)
classes, counts = np.unique(y_train_raw, return_counts=True)
max_count = counts.max()

X_balanced, y_balanced = [], []
for c in classes:
    X_c = X_train_raw[y_train_raw == c]
    y_c = y_train_raw[y_train_raw == c]
    idx = rng.choice(len(X_c), size=max_count, replace=True)
    X_balanced.append(X_c[idx])
    y_balanced.append(y_c[idx])

X_balanced = np.vstack(X_balanced)
y_balanced = np.hstack(y_balanced)

# Train/Validation split
x_train, x_val, y_train, y_val = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

# Reshape & one-hot encode labels
x_train = x_train.reshape(-1, 187, 1)
x_val   = x_val.reshape(-1, 187, 1)
X_test  = X_test.reshape(-1, 187, 1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_val   = tf.keras.utils.to_categorical(y_val, num_classes=5)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes=5)

# CNN Model
model = Sequential([
    Input(shape=(187, 1)),
    Conv1D(64, 6, activation='relu'), BatchNormalization(), MaxPool1D(3, 2, padding="same"),
    Conv1D(64, 3, activation='relu'), BatchNormalization(), MaxPool1D(2, 2, padding="same"),
    Conv1D(64, 3, activation='relu'), BatchNormalization(), MaxPool1D(2, 2, padding="same"),
    LSTM(64, return_sequences=True), LSTM(32),
    Flatten(),
    Dense(64, activation='relu'), Dense(32, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

# Train
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=10, batch_size=32, callbacks=callbacks, verbose=1)

# Predictions
y_pred_labels = np.argmax(model.predict(X_test), axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Classification report
print(classification_report(y_true_labels, y_pred_labels, target_names=[
    'Normal', 'Atrial Premature', 'PVC', 'Fusion V-N', 'Fusion P-N'
]))

# Confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, normalize='true')
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=[
    'Normal', 'Atrial Premature', 'PVC', 'Fusion V-N', 'Fusion P-N'
], yticklabels=[
    'Normal', 'Atrial Premature', 'PVC', 'Fusion V-N', 'Fusion P-N'
])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix")
plt.show()
