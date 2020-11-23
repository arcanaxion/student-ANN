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
file_path = os.path.join(dir_path, "student-mat.csv")

features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'absences', 'G3', 'famsize', 'sex']
label = ['address']
columns = features + label
student = pd.read_csv(file_path, header=0, usecols=columns)

# Check for null values
print("Null value check:")
print(student.isnull().sum())

# Print head of student dataframe
print("Print 'student' head:")
print(student.head())

# Binarise
student['famsize'] = (student['famsize'] == 'GT3').astype(int)
student['sex'] = (student['sex'] == 'M').astype(int) 


# Describe
print(student.describe().transpose())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(student.drop(label, axis=1), student[label], train_size=0.8)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(pd.DataFrame(X_train, columns=features).describe().transpose())

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(2), max_iter=1000)

mlp.partial_fit(X_train, y_train.values.ravel(), student[label[0]].unique())
visualise(mlp)

mlp.fit(X_train, y_train.values.ravel())
visualise(mlp)

predictions = mlp.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

plt.show()