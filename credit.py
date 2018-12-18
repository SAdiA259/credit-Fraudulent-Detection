import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from numpy import genfromtxt
import sklearn
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

data = pd.read_csv("creditcard.csv")

#print(data.head())
## find null values
#A = data.isna().sum()
#print (A)
## finding class distribution
#sns.countplot(x='class', data=data, palette='hls')
#plt.show()
#data['class'].value_counts()

## balancing data set
#over_sampling= SMOTE(random_state=123)
#data=over_sampling.fit_sample(data)
#print(np.bincount(data))

###########################################################
## set parameters

#np.set_printoptions(threshold=np.inf) #print all values in numpy array

###########################################################
#parameters

learning_rate = 0.01
n_epochs = 27000  #1000
batch_size = 100
#display_step = 1

## a smarter learning rate for gradient optimizer
#learningRate = tf.train.exponential_decay(learning_rate=0.0008,
#                                          global_step=1,
#                                          decay_steps=trainX.shape[0],
#                                          decay_rate=0.95,
#                                          staircase=True)


###########################################################

# Convert to one hot data
def convertOneHot_data2(data):
    y=np.array([int(i) for i in data])
    #print y[:20]
    rows = len(y)
    columns = y.max() + 1
    a = np.zeros(shape=(rows,columns))
    #print a[:20,:]
    print rows
    print columns
    #rr = raw_input()
    #y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        #y_onehot[i]=np.array([0]*(y.max() + 1) )
        #y_onehot[i][j]=1
        a[i][j]=1
    return (a)

#############################################################
## manual 10 fold crossvalidation get sets

def select_fold_to_use_rc(X, y, k):
    K = 10
    num_samples = len(X[:,0])
    #print num_samples
    #rr = raw_input()
    #for i, x in enumerate(X):
    #    print "i", str(i)
    #    print "k", str(k)
    #    print " i % K != k"   
    #    print i % K
    #    print i % K != k 
    #    rr = raw_input()
    training_indices = [i for i, x in enumerate(X) if i % K != k]
    testing_indices = [i for i, x in enumerate(X) if i % K == k]
    #print training_indices
    #print testing_indices
    #rr = raw_input()
    X_train = X[training_indices]
    y_train = y[training_indices]
    X_test  = X[testing_indices]
    y_test  = y[testing_indices]		
    return X_train, X_test, y_train, y_test

##################################
#f_numpy = open("creditcard.csv",'r')
#data = np.loadtxt(f_numpy, delimiter=",", skiprows=1)

#data = np.genfromtxt('creditcard.csv', delimiter=',', skip_header=1) 
### shape of data 
a= data.shape
print(a)
#print data
x = data.iloc[:, :29]
y = data.iloc[:, 30:]
over_sampling = SMOTE(random_state=123)
x,y=over_sampling.fit_sample(x,y)
data =data.sample(frac=1)

#print (y)


############################################################################################
## % train test split


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)


############################################################################################
############################################################
## feature scalinng

scaler = MinMaxScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

###########################################################

print "starting to convert data to one hot encoding"


y_train_onehot = pd.get_dummies(y_train)
y_test_onehot = pd.get_dummies(y_test)

#X_train_std= np.asanyarray(X_train_std.values, dtype="float32")
#X_test_std= np.asanyarray(X_test_std.values, dtype="float32")
y_train_onehot= np.asanyarray(y_train_onehot.values, dtype="float32")
y_test_onehot= np.asanyarray(y_test_onehot.values, dtype="float32")

#y_train_onehot = convertOneHot_data2(y_train) 

#y_test_onehot = convertOneHot_data2(y_test) 


#print y_train_onehot[:20,:]
#rr = raw_input()

print "data has been loaded from csv"

###########################################################
# features (A) and classes (B)
#  A number of features, 784 in this example
#  B = number of classes, 10 numbers for mnist (0,1,2,3,4,5,6,7,8,9)


A = X_train_std.shape[1]   #num features
B = y_train_onehot.shape[1]   #num classes
samples_in_train = X_train_std.shape[0]
samples_in_test = X_test_std.shape[0]
#A = len(X_train[0,:])  # Number of features
#B = len(y_train_onehot[0]) #num classes
print "num features", A
print "num classes", B
print "num samples train", samples_in_train
print "num samples test", samples_in_test
print "press enter"
rr = raw_input()

###################################################
## print stats 
precision_scores_list = []
accuracy_scores_list = []

def print_stats_metrics(y_test, y_pred):    
    print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )
    #Accuracy: 0.84
    accuracy_scores_list.append(accuracy_score(y_test,   y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print "confusion matrix"
    print(confmat)
    print pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

#####################################################################

def plot_metric_per_epoch():
    x_epochs = []
    y_epochs = [] 
    for i, val in enumerate(precision_scores_list):
        x_epochs.append(i)
        y_epochs.append(val)
    
    plt.scatter(x_epochs, y_epochs,s=50,c='lightgreen', marker='s', label='score')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('Score per epoch')
    plt.legend()
    plt.grid()
    plt.show()


#####################################################################

def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=w_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)

##########################################################
#defines network architecture
#deep neural network with 4 hidden layers

def inference_deep_4layers(x_tf, A, B):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x_tf, [A, 21],[21])
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [21, 21],[21])
    with tf.variable_scope("hidden_3"):
        hidden_3 = layer(hidden_2, [21, 21],[21])
    with tf.variable_scope("hidden_4"):
        hidden_4 = layer(hidden_3, [21, 21],[21])
    with tf.variable_scope("output"):
        output = layer(hidden_4, [21, B], [B])
    return output

##########################################################

def loss_deep(output, y_tf):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y_tf)
    
    loss = tf.reduce_mean(xentropy) 
    return loss
    
###########################################################

def training(cost):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #train_op = optimizer.minimize(cost)
    optimizer= tf.train.AdamOptimizer(0.005).minimize(cost)
    return optimizer

###########################################################
## add accuracy checking nodes

def evaluate(output, y_tf):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_tf,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


###########################################################

x_tf = tf.placeholder("float", [None, A]) # Features
y_tf = tf.placeholder("float", [None,B]) #correct label for x

###############################################################

output = inference_deep_4layers(x_tf, A, B) ## for deep NN with 2 hidden layers
cost = loss_deep(output, y_tf)
train_op = training(cost)
eval_op = evaluate(output, y_tf)


##################################################################
## for metrics

y_p_metrics = tf.argmax(output, 1)

##################################################################
# Initialize and run

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

##################################################################
#batch size is 100

num_samples_train_set = X_train_std.shape[0] #len(X_train[:,0]) 
#num_samples_train_set = len(X_train[:,0])
num_batches = int(num_samples_train_set/batch_size)

##################################################################


print "starting training and testing"
print("...")
# Run the training
final_result = ""
for i in range(n_epochs):
    print "epoch %s out of %s" % (i, n_epochs)
    for batch_n in range(num_batches):
        sta = batch_n*batch_size
        end = sta + batch_size
        sess.run(train_op, feed_dict={x_tf: X_train_std[sta:end,:],
                                            y_tf: y_train_onehot[sta:end,:]}) 
    
    print "-------------------------------------------------------------------------------"    
    print "Accuracy score"
    result, y_result_metrics = sess.run([eval_op, y_p_metrics], feed_dict={x_tf: X_test_std,
                                          y_tf: y_test_onehot})
    print "Run {},{}".format(i,result)
    #print final_result
    y_true = np.argmax(y_test_onehot,1)
    print_stats_metrics(y_true, y_result_metrics)
    if i == 1000:
        plot_metric_per_epoch()



##################################################################

print "<<<<<<DONE>>>>>>"
