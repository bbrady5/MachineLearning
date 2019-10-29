import numpy 
import pandas 
import time
import os
 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



def load_data(filename):
    #navigate to the folder which holds both the python script and the 4 files
    my_folder = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(my_folder, filename)
    f = open(my_file, 'rt')
    x = f.readlines()
    f.close()
    features = list(map(list, x)) #make each bit a separate feature
    for item in features:
        item.remove("\t") #remove any tabs
    for item in features:
        item.remove("\n") #remove any newlines
    nparr = numpy.array(features) #convert to numpy array
    nparr = nparr.astype(int) #change data type to int so that the classes 0 and 1 can be ascertained
    print(nparr.shape)
    return nparr
def load_label(filename):
    #navigate to the folder which holds both the python script and the 4 files
    my_folder = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(my_folder, filename)
    f = open(my_file, 'rt')
    x = f.readlines()
    f.close()
    features = list(map(list, x)) #make each bit a separate feature
    for item in features:
        item.remove("\n") #remove any newlines
    nparr = numpy.array(features) #convert to numpy array
    nparr = nparr.astype(int) #change data type to int so that the classes 0 and 1 can be ascertained
    print(nparr.shape)
    return nparr

def main():
    print("Importing data...")
    training_data = load_data("training_data")
    training_label = load_label("training_label")
    testing_data = load_data("testing_data")
    testing_label = load_label("testing_label")
    print("------------------")
    
    index = 0 
    mean_accuracy = 0
    mean_precision = 0
    mean_recall = 0
    num_models = 32
    while index<32: 
        print("Model for Bit", (31-index)) # MSB = 31 (left), LSB = 0 (right)
    
        
        X_train = training_data
        y_train = training_label[:, index]
       
        X_test = testing_data
        y_test = testing_label[:, index]
        
        
        clf = MLPClassifier(hidden_layer_sizes=(100,10), activation='logistic', solver='adam', max_iter=500, random_state=1)
        
        clf.fit(X_train, y_train)
        
        found_classes = clf.classes_
        print("Classes:", found_classes)
        
        if len(found_classes) == 1:
            print("Only one class determined for the training set - Therefore predict all output for bit", (31-index), "to be", found_classes)
            num_models -= 1 #decrement so that mean accuracy, mean precision, and mean recall are divided by the correct number of predictive models
        else:
            y_pred = clf.predict(X_test) 
            
            #Confusion matrix for this model
            print("Confusion Matrix: \n", 
            confusion_matrix(y_test, y_pred, labels=[0,1])) 
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
            print('TN: ', tn,'FP: ', fp, 'FN: ', fn, 'TP: ', tp)
            
            #Accuracy for this model
            a = accuracy_score(y_test, y_pred)*100
            print("Accuracy : ", a, "%") 
            mean_accuracy += a
            
            #Precision for this model
            b = precision_score(y_test, y_pred, average='binary') #average='binary' returns the precision for the positive label (1 - the class that denotes errors)
            print("Precision : ", b) 
            mean_precision += b
            
            #Recall for this model
            c = recall_score(y_test, y_pred, average='binary') #average='binary' returns the recall for the positive label (1 - the class that denotes errors)
            print("Recall : ", c) 
            mean_recall += c
            
        print("------------------")
        index += 1
        
    
    aa = mean_accuracy/num_models
    print("Mean Accuracy for all Bits: ", aa, "%")
    
    bb = mean_precision/num_models
    print("Mean Precision for all Bits: ", bb)
    
    cc = mean_recall/num_models
    print("Mean Recall for all Bits: ", cc)
    
    
start_time = time.time()
main()
end_time = time.time() - start_time
#print("Running time of program", time.time()-start_time, 'seconds.')
print('Running time of program:  {0:0.2f} seconds'.format(end_time), '({0:0.2f} minutes)'.format(end_time/60))
     
