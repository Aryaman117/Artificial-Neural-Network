# Artificial Neural Network

This project implements an Artificial Neural Network (ANN) to predict customer churn from a bank's customer dataset. The workflow includes data preprocessing, building and training the model, and evaluating its performance.

## Part 1 - Data Preprocessing

### Importing the libraries
We start by importing the necessary libraries:
```python
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__
```

### Importing the dataset
The dataset is loaded, and the feature matrix `X` and the target variable `y` are extracted:
```python
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
```

### Encoding categorical data
- **Label Encoding the "Gender" column:**
  The "Gender" column is encoded to numerical values.
  ```python
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  X[:, 2] = le.fit_transform(X[:, 2])
  print(X)
  ```

- **One Hot Encoding the "Geography" column:**
  The "Geography" column is one-hot encoded to avoid ordinal relationships.
  ```python
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import OneHotEncoder
  ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
  X = np.array(ct.fit_transform(X))
  print(X)
  ```

### Splitting the dataset into the Training set and Test set
The dataset is split into training and test sets to evaluate the model performance on unseen data:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Feature Scaling
Feature scaling is applied to ensure all features contribute equally to the distance calculations in the ANN:
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Part 2 - Building the ANN

### Initializing the ANN
A Sequential model is initialized:
```python
ann = tf.keras.models.Sequential()
```

### Adding the input layer and the first hidden layer
The first hidden layer with 6 neurons and ReLU activation function is added:
```python
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
```

### Adding the second hidden layer
A second hidden layer with 6 neurons and ReLU activation function is added:
```python
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
```

### Adding the output layer
The output layer with 1 neuron and sigmoid activation function is added to predict the binary outcome:
```python
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

## Part 3 - Training the ANN

### Compiling the ANN
The ANN is compiled with Adam optimizer and binary cross-entropy loss function:
```python
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Training the ANN on the Training set
The model is trained on the training set for 100 epochs with a batch size of 32:
```python
ann.fit(X_train, y_train, batch_size=32, epochs=100)
```

## Part 4 - Making Predictions and Evaluating the Model

### Predicting the result of a single observation
We use the trained ANN to predict if a specific customer will leave the bank:
```python
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
```
The ANN predicts that this customer will stay with the bank.

**Important Notes:**
1. The input values are wrapped in double square brackets to match the 2D array input requirement of the `predict` method.
2. The "Geography" field is one-hot encoded, with "France" represented as `[1, 0, 0]`.

### Predicting the Test set results
The ANN's predictions on the test set are compared with the actual outcomes:
```python
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
```

### Making the Confusion Matrix
The confusion matrix and accuracy score are calculated to evaluate the model's performance:
```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```

The confusion matrix provides a summary of prediction results on the test set:
- **True Negatives (TN):** Number of correct predictions where the customer did not leave.
- **False Positives (FP):** Number of incorrect predictions where the customer did not leave but was predicted to leave.
- **False Negatives (FN):** Number of incorrect predictions where the customer left but was predicted to stay.
- **True Positives (TP):** Number of correct predictions where the customer left.

The accuracy score gives the proportion of correctly predicted instances among the total instances evaluated.
