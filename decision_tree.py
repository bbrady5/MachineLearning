import numpy 
import pandas 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


#importing data

filename = 'training_data'
raw_data0 = open(filename, 'rt')
training_data = numpy.loadtxt(raw_data0, delimiter=",")
print(training_data.shape)
#training_data.head(4)


filename = 'training_label'
raw_data1 = open(filename, 'rt')
training_label = numpy.loadtxt(raw_data1, delimiter=",")
print(training_label.shape)


filename = 'testing_data'

raw_data2 = open(filename, 'rt')
testing_data = numpy.loadtxt(raw_data2, delimiter=",")
print(testing_data.shape)

filename = 'testing_label'
raw_data3 = open(filename, 'rt')
testing_label = numpy.loadtxt(raw_data3, delimiter=",")
print(testing_label.shape)




list =[]

bit = 0

while bit < 32:
    print("Creating 32 models: Using Classifier of BIT " + str(bit))
    
    print(training_data)
    print(training_label[:, bit])
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(training_data, training_label[:, bit])
    
    # running this classifier for all bits
    index = 0
    total_mean_accuracy = 0
    while index<32:
        print("predictions for bit:" + str(index))
        y_pred = clf.predict(testing_data) 
        
        print("Actual values:")
        print(testing_label[:, index])
        print("Predicted values:") 
        print(y_pred) 
        
        print("Confusion Matrix: ", 
        confusion_matrix(testing_label[:, index], y_pred)) 
            
        print ("Accuracy : ") 
        a = accuracy_score(testing_label[:, index], y_pred)*100
        print(a) 
        total_mean_accuracy += a
        #print("Report : ", 
        #classification_report(testing_label[:, index2], y_pred)) 
        
        index += 1
        
    
    ab = total_mean_accuracy/32
    list.append(ab)
    print("total mean accuracy:" + str(ab))

    bit += 1
   

print(list)
print(len(list))
print(max(list))  
print(list.index(min(list)))
    
    
'''
    y_pred = clf.predict(testing_data) 
    #print(testing_data[:, [13, 29]])
    print("Actual values:")
    print(testing_label[:, bit])
    print("Predicted values:") 
    print(y_pred) 
        
    print("Confusion Matrix: ", 
    confusion_matrix(testing_label[:, bit], y_pred)) 
            
    print ("Accuracy : ") 
    a = accuracy_score(testing_label[:, bit], y_pred)*100
    print(a) 
    total_mean_accuracy += a
    ab = total_mean_accuracy/32
    list.append(ab)
    print("total mean accuracy:" + str(ab))
    
    total_mean_accuracy = 0
    ab = 0
    bit +=1

print(list)
print(max(list))
'''

'''
input1=13
input2=29
label=29
clf = DecisionTreeClassifier(random_state=0)
clf.fit(training_data[:, [input1, input2]], training_label[:, label])

index = 0
index2 = index + 16
total_mean_accuracy = 0
while index<16:
    print("predictions for bits:" + str(index))
    y_pred = clf.predict(testing_data[:, [index, index2]]) 
    #print(testing_data[:, [13, 29]])
    print("Actual values:")
    print(testing_label[:, index2])
    print("Predicted values:") 
    print(y_pred) 
    
    print("Confusion Matrix: ", 
    confusion_matrix(testing_label[:, index2], y_pred)) 
        
    print ("Accuracy : ") 
    a = accuracy_score(testing_label[:, index2], y_pred)*100
    print(a) 
    total_mean_accuracy += a
    print("Report : ", 
    classification_report(testing_label[:, index2], y_pred)) 
    
    index = index+1
    index2 +=1
print("total mean accuracy:" + str(total_mean_accuracy/32))




'''




'''
# Function importing Dataset 
def importdata(): 
    balance_data = pd.read_csv( 'https://www.dropbox.com/s/n2m6dzkvmyu33ii/training_data?dl=0', sep= '\t', header = None) 
    
    # Printing the dataset shape 
    print ("Dataset Length: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
    
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 

importdata()
print("done")
'''

'''
# Function to split the dataset 
def splitdataset(balance_data): 

    # Seperating the target variable 
    X = balance_data.values[:, 1:5] 
    Y = balance_data.values[:, 0] 

    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100) 
    
    return X, Y, X_train, X_test, y_train, y_test 
    
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 

    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 

    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
    
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 

    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 

    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 


# Function to make predictions 
def prediction(X_test, clf_object): 

    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
    
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
    
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
    
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
    
    print("Report : ", 
    classification_report(y_test, y_pred)) 

# Driver code 
def main(): 
    
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
    
    # Operational Phase 
    print("Results Using Gini Index:") 
    
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
    
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    
# Calling main function 
if __name__=="__main__": 
    main() 
    
    
    





'''