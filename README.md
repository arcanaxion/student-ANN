# Predicting Students' Alcohol Consuption Using MLP Classifier

# NOTE

The content of this `README` is extracted from excerpts of my report. Kindly drop me an email should you be interested to read the full report.

# Purpose
This study uses key demographic and and societal factors of interest to predict a student’s alcohol consumption on weekdays, represented on a scale from 1 to 5.

The dataset is sourced from Kaggle [1]. The estimator used is the MLP Classifier from scikit-learn.

# Features
7 input features and 1 target (`Dalc`) are described in the following table:
| Name | Description | Type|
|---|---|---|
|Sex	| Student’s sex (Male or Female) |	Binary
Age	| Student’s age	| Numeric
Activities | Student’s participation in extra-curricular activities (Yes or No) | Binary
Free time |	Free time after school (1 - very low to 5 - very high) |	Ordinal
Go out |	Going out with friends (1 - very low to 5 - very high) |	Ordinal
Address |	Home address type (Urban or Rural) |	Binary
Absences |	Number of school absences	| Numeric
Dalc |	Workday alcohol consumption (1 - very low to 5 - very high)	| Ordinal

# Architecture

MLPClassifier from the scikit-learn library uses a feed-forward architecture that trains using backpropagation [2]. The classifier also intrinsically uses Softmax as the output function for multi-class classification which predicts the probability that a particular output neuron or class is the actual output. The class with the highest probability is considered the predicted output. This makes it compatible with our purpose of performing classification with 5 different classes.

The specific way that the MLPClassifier functions is generally hidden or abstracted to the designer and some control such as using a different loss function for error calculation is not provided — the model is fixed to using a log-loss function based on LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm) or stochastic gradient descent. However, many parameters are still accessible for setting up and tuning the neural network.

One of the first parameters we decide is the number of hidden layers and the number of hidden neurons for each hidden layer. In solving simple problems without the need of ‘deep learning’ for applications such as computer vision, generally, a single hidden layer is suggested [4]. Furthermore, [4] suggests that the number of hidden neurons used should be between the number of input neurons and output neurons which are 7 and 5 respectively in our case, making the suggested number of hidden neurons be 6. Besides that, [5] suggests a geometric pyramid rule to determine the number of hidden neurons to use, which follows the formula:

n<sub>h</sub> = √(n<sub>i</sub>n<sub>o</sub>)

In the aforementioned formula, n<sub>h</sub>, n<sub>i</sub> and n<sub>o</sub> are the number of hidden, input and output neurons respectively. Accordingly, the resultant value for n<sub>h</sub> is 5.9161 which is rounded up to 6. Therefore, with 2 methods suggesting the use of 6 hidden neurons, we proceed with a model of 1 hidden layer with 6 hidden neurons.

The solver for weight optimization used is LBFGS is an optimizer in the family of quasi-Newton methods. The documentation for MLPClassifier [3] suggests that for smaller datasets LGFGS converges faster and performs better.

The activation function used is called ‘relu’ or rectified linear unit function and returns f(x) = max(0, x), which in simpler terms returns the weighted input but caps it at 0 if it is less than 0. Relu is the default activation function and is used in most modern approaches.

Finally, the maximum number of iterations is set to 1000 because MLPClassifier’s default of 200 tends to be reached quickly, causing the model to converge prematurely.

# Results

Classification Matrix:

|          |     **1**     |     **2**    |     **3**    |     **4**    |     **5**    |
|----------|-----------|----------|----------|----------|----------|
|     **1**    |     88    |     4    |     0    |     1    |     0    |
|     **2**    |     17    |     6    |     0    |     0    |     0    |
|     **3**    |     8     |     3    |     1    |     0    |     1    |
|     **4**    |     1     |     1    |     1    |     0    |     0    |
|     **5**    |     0     |     1    |     0    |     0    |     0    |

Accuracy: 0.7143

# Discussion

Despite the high mean accuracy of 0.7143, we notice that the neural networks performance was not actually that good. The target output ‘Dalc’ has mostly low values, and more than 50% are simply 1.

Observing the classification matrix, we can see that the system predicted class 1 a large amount of the time. When misclassifications occurred, they were mostly misclassified as class 1. That is why despite a high sensitivity for class 1 at 0.95, its precision is only 0.77. That is because the system made many predictions as class 1, including for many data records that were not class 1.

It was still able to accurately predict some of the data that belonged in classes 2 and 3, scoring 26% sensitivity (recall) for class 2 and 8% for class 3. However, for classes 4 and 5 which had only a meagre 3 and 1 data records respectively in the expected output, the system did not get a single accurate prediction.

Since so many of the data records belonged in class 1 anyway, the model gives a slightly misleading impression with its high accuracy of 0.7143.

Performance could improve if the dataset was substantially larger. As seen in the classification matrix, the testing data had only 4 data records in total for the final two classes. Extrapolating from this, this means that there were only about 20 data records in total that belonged to the final two classes. The neural network would not have been able to train much and adapt to these significantly fewer occurring cases.

# References

[1]	U. C. I. M. Learning, “Student Alcohol Consumption,” Kaggle, 19-Oct-2016. [Online]. Available: https://www.kaggle.com/uciml/student-alcohol-consumption. [Accessed: 25-Nov-2020].

[2]	“1.17. Neural network models (supervised),” scikit. [Online]. Available: https://scikit-learn.org/stable/modules/neural_networks_supervised.html. [Accessed: 26-Nov-2020].

[3]	“sklearn.neural_network.MLPClassifier,” scikit. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html. [Accessed: 26-Nov-2020].

[4]	J. Heaton, “The Number of Hidden Layers,” Heaton Research, 20-Jul-2020. [Online]. Available: https://www.heatonresearch.com/2017/06/01/hidden-layers.html. [Accessed: 26-Nov-2020].

[5]	T. Masters, Practical neural network recipes in C++. San Diego: Academic Press, 1999.
