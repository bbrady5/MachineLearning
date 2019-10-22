import numpy 
import pandas 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 



def load_data(filename):
    f = open(filename, 'rt')
    x = f.readlines()
    f.close()
    test = list(map(list, x))
    for item in test:
        item.remove("\t")
    for item in test:
        item.remove("\n")
    nparr = numpy.array(test)
    nparr = nparr.astype(int)
    print(nparr.shape)
    return nparr
def load_label(filename):
    f = open(filename, 'rt')
    x = f.readlines()
    f.close()
    test = list(map(list, x))
    for item in test:
        item.remove("\n")
    nparr = numpy.array(test)
    nparr = nparr.astype(int)
    print(nparr.shape)
    return nparr

training_data = load_data("training_data")
training_label = load_label("training_label")
testing_data = load_data("testing_data")
testing_label = load_label("testing_label")




list =[]

bit = 31

print("Creating 32 models: Using Classifier of BIT " + str(bit))

print(training_data)
print(training_label[:, bit])
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
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
    print("Report : ", 
    classification_report(testing_label[:, index], y_pred)) 
    
    index += 1
    

ab = total_mean_accuracy/32
list.append(ab)
print("total mean accuracy:" + str(ab))


   

print(list)
print(len(list))
print(max(list))  
print(list.index(min(list)))
    
 
