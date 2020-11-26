import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt

def visualise(mlp):
  # get number of neurons in each layer
  n_neurons = [len(layer) for layer in mlp.coefs_]
  n_neurons.append(mlp.n_outputs_)

  # calculate the coordinates of each neuron on the graph
  y_range = [0, max(n_neurons)]
  x_range = [0, len(n_neurons)]
  loc_neurons = [[[l, (n+1)*(y_range[1]/(layer+1))] for n in range(layer)] for l,layer in enumerate(n_neurons)]
  x_neurons = [x for layer in loc_neurons for x,y in layer]
  y_neurons = [y for layer in loc_neurons for x,y in layer]

  # identify the range of weights
  weight_range = [min([layer.min() for layer in mlp.coefs_]), max([layer.max() for layer in mlp.coefs_])]

  # prepare the figure
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  # draw the neurons
  ax.scatter(x_neurons, y_neurons, s=100, zorder=5)
  # draw the connections with line width corresponds to the weight of the connection
  for l,layer in enumerate(mlp.coefs_):
    for i,neuron in enumerate(layer):
      for j,w in enumerate(neuron):
        ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'white', linewidth=((w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)*1.2)
        ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'grey', linewidth=(w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
math_path = os.path.join(dir_path, "student-mat.csv")
port_path = os.path.join(dir_path, "student-por.csv")

math_student = pd.read_csv(math_path, header=0)
port_student = pd.read_csv(port_path, header=0)

# Drop duplicates based on specific columns as described on the Kaggle dataset
from_kaggle = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]
student = pd.concat([math_student, port_student]).drop_duplicates(subset=from_kaggle).reset_index(drop=True)

features = ['sex', 'age', 'activities', 'freetime', 'goout', 'address', 'absences']
label = ['Dalc']
columns = features + label

student = student[columns]

# Check for null values
print("Null value check:")
print(student.isnull().sum())

# Print head of student dataframe
print("Print 'student' head:")
print(student.head())

# Binarise to 1 and 0
student['sex'] = (student['sex'] == 'M').astype(int) 
student['activities'] = (student['activities'] == 'yes').astype(int) 
student['address'] = (student['address'] == 'U').astype(int) 

# Describe
print(student.describe().transpose())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(student[features], student[label], train_size=0.8, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(pd.DataFrame(X_train, columns=features).describe().transpose())

# 'lbfgs' solver for weight optimizer is suggested as a more efficient
# and accurate solver for small datasets
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(6), max_iter=1000, activation='relu', solver='lbfgs', random_state=1)

# Cannot be used with 'lbfgs' solver for weight optimizer
# mlp.partial_fit(X_train, y_train.values.ravel(), student[label[0]].unique())
# visualise(mlp)

mlp.fit(X_train, y_train.values.ravel())
visualise(mlp)

predictions = mlp.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification report:")
print(classification_report(y_test, predictions))

print("Mean accuracy:", mlp.score(X_test, y_test))
print("Log-loss function:", mlp.loss_)

print("Number of iterations:", mlp.n_iter_)

# plt.show()