
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer()


x = cancer.data
y = cancer.target

# Create test and training sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)


classes = ['malignant', 'benign']

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy import io
import numpy as np
import tkinter as tk
from tkinter import filedialog

cancer = io.loadmat('breast_cancer_data.mat')
X = cancer['data']
y = cancer['target'].flatten()


X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)


split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

# Define what the values mean
classes = ['malignant', 'benign']

# Create a Sequential model using Keras
model = Sequential([
    Dense(units=1, input_dim=X_train.shape[1], activation='linear')
])

# Compile the model with a custom loss function for SVM
def svm_loss(y_true, y_pred):
    hinge_loss = tf.maximum(0., 1. - y_true * y_pred)
    return tf.reduce_mean(hinge_loss)

model.compile(optimizer='adam', loss=svm_loss)

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=0)


loss = model.evaluate(X_test, y_test)
print(f'Model Loss on Test Set: {loss:.4f}')


y_pred = model.predict(X_test).flatten()

y_pred_labels = np.sign(y_pred)

accuracy = np.mean(y_pred_labels == y_test)
print(f"Accuracy: {accuracy:.4f}")


window = tk.Tk()
window.title("Breast Cancer Classification with SVM")
window.geometry("800x600")


def choose_file():
    file_path = filedialog.askopenfilename()
    print(f'Selected file: {file_path}')


file_button = tk.Button(window, text="Choose File", command=choose_file)
file_button.pack()

window.mainloop()
