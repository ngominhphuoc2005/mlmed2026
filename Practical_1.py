import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, MaxPool1D, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Download latest version
path = kagglehub.dataset_download("shayanfazeli/heartbeat")

train_path = os.path.join(path, "mitbih_train.csv")
test_path = os.path.join(path, "mitbih_test.csv")
train = pd.read_csv(train_path, header=None)
test  = pd.read_csv(test_path, header=None)

X_train_raw = train.iloc[:, :-1].values
y_train_raw = train.iloc[:, -1].astype(int).values
X_test      = test.iloc[:, :-1].values
y_test      = test.iloc[:, -1].astype(int).values

ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X_train_raw, y_train_raw)

x_train, x_val, y_train, y_val = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

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
    LSTM(64, return_sequences=True), 
    LSTM(32),
    Dense(64, activation='relu'), Dense(32, activation='relu'),
    Dense(5, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
]
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=30, batch_size=32, callbacks=callbacks, verbose=1)

# Result
y_pred_labels = np.argmax(model.predict(X_test), axis=1)
y_true_labels = np.argmax(y_test, axis=1)
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
