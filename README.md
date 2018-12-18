# credit-Fraudulent-Detection
Predicting if a credit transaction is Legit or Fraud


Abstract:

Due to the increase in credit frauds and with large amount of data and numerous credit channels and transactions It has been a challenging task to predict and catch credit fraudulent transaction. This paper discusses a predictive model to classify fraudulent and non-fraudulent credit transactions using Machine learning approach. A model is built with logistic regression deep neural network in TensorFlow to predict if a transaction is fraudulent or not.

Problem Statement

	Credit card fraud is on a breakneck incline. Crooks are using ever more sophisticated methods beyond the good ole’ lost or stolen, mail intercept, phone scam or email phishing to the current more high-tech card skimmers and WiFi hotspot traffic packet intercepts.
14.2 million
 credit card numbers exposed in 2017.
 A stagering 88% increase from the previous year in 2016. [2]
So, what is the cost and loss of credit card fraud? Is the cost and loss solely monetary? Who are the victims and how common is it?
The following statistics according to recent studies by the Identity Theft Resource Center (ITRC), stated credit card numbers exposed in 2017 totaled 14.2 million, which is up 88% over the previous year. A single year increase of 88% is staggering and practically unfathomable at any capacity.
Credit card fraud was the most common form of identity theft (133,015), in 2017 according to the Federal Trade Commission. [3] 

Introduction

Banking and financial institutions are experiencing intense difficulties due to fraudulent transactions. There was a $21.84 billion fraud loss on credit, debit and prepaid cards in 2015 alone. An article published in Forbes predicts that this could grow by 45% by 2020. Usage of traditional authentication and validation rules in longer an effective method to catch fraudulent transactions because of different payment methods, lenders, cards and huge customer data. Machine learning is an efficient approach to learn user’s behavior and purchase behavior over time with massive amount of data in large volume and velocity. Using machine learning for fraud detection is no longer a trend, it is a necessity

Procedure:

The data set is very imbalanced and SMOTE (Synthetic Monitory Over Sampling Technique) is used to perform over sampling of minority class. Logistic regression is used in TensorFlow to perform classification of fraudulent transaction. TensorFlow is a good platform to perform large-data analysis with GPU. The layout of our model consists of following functions:
1. Inference: calculates the equation
     y= Wx+b and returns the predicted output.
2. Cost function: calculates the cost equation for logistic regression
3. Train: minimizes cost function using Gradient descent algorithm.
4. Evaluation: comparing the actual and predicted output.

One Hot Encoding
One hot encoding is used for categorical data. In our case 0 - Non Fraudulent, 1- Fraudulent. Some algorithms can work with label data directly, while others need data transformation (converting to numerical form) before applying them to algorithms.we applied one hot coding to our classes(categories) for this purpose.

Feature Scaling
Minmax scaler is used to scale the features, the properties of normal standard distribution with mean=0 and standard deviation=1. We used the standardized trained features and applied them to the model.
Learning rate
Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient. The lower the value, the slower we travel along the downward slope. Learning rate of 0.01 is used to optimize the cost function in the model.

Cost Function
Here we define the structure of LR in inference, in Deep Neural Network. The cost equation for LR is as below:
s(z) =  1 / 1+ e-z
In this function we define W(weight) and b(bias) for LR and we use softmax to get a clear vision of probability of belonging the data to different classes. The goal is to reduce the cost in Gradient Descent. Using learning rate=0.001, AdamOptimizer is used as an optimizer, and optimizer.minimize(cost) is used to minimize the cost associated with this algorithm.

Activation Function
The Sigmoid Function is mostly used when implementing Logistic Regression but since we are making Deep Neural Network, ReLu seems to be a better activation function for hidden layers and it was our choice. The output of the regression is calculated as:
Y= θ0 + θ1x1 + θ1x2
To get the output values as [0,1],sigmoid function is applied.
 
Network architecture
Deep neural network is used with logistic regression model, with Number of hidden layers= 4 and ultimately output with 21 features.

Batch
Batch processing is used when dealing with large amount of data. Dividing data intro
batches and run them in each epoch. In this experiment we implemented batching with the size of 100 to get better performance of the model.

Epoch
An epoch is one full training cycle. After every sample in the data is seen it starts from the beginning of data set and it is the next epoch now. At the end of each epoch batch updates itself and this step is repeated until the number of epochs is finished. Number of epochs chose for the experiment is 27000 epochs to increase the model’s performance.

Evaluation Metrics:
Accuracy     Batch    Epoch    N.of layer      Learning rate
0.99         100       27000       5                 0.001
Precision: 0.997
F Score: 0.985
Recall: 0.973 


