import numpy 
import pandas 
import time
#from sklearn.linear_model import LogisticRegression 
#from sklearn.neural_network import MLPClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#from sklearn.metrics import classification_report 



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

def main():
    print("Importing data...")
    training_data = load_data("training_data")
    training_label = load_label("training_label")
    testing_data = load_data("testing_data")
    testing_label = load_label("testing_label")
    
    index = 0 # ignore class for bit 0 because there is only one class (0)
    mean_accuracy = 0
    mean_precision = 0
    mean_recall = 0
    
    while index<32: #31
        print("Model for Bit " + str(31 - index)) # MSB = 31 (left), LSB = 0 (right)
        
        #print(training_data)
        #print(training_label[:, index])
        
        #clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        #clf = RandomForestClassifier(n_estimators=100, max_depth=2, bootstrap=False, random_state=0)
        #clf = svm.SVC(gamma='scale')
        #clf = KNeighborsClassifier(n_neighbors=3)
        #clf = GaussianNB()
        #clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=500, shuffle=True)
        clf = BernoulliNB(alpha = 0.01)
        #clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(training_data, training_label[:, index])
        
        y_pred = clf.predict(testing_data) 
        
        print("Actual values:")
        print(testing_label[:, index])
        print("Predicted values:") 
        print(y_pred) 
        
        print("Confusion Matrix: \n", 
        confusion_matrix(testing_label[:, index], y_pred, labels=[0,1])) 
        tn, fp, fn, tp = confusion_matrix(testing_label[:, index], y_pred, labels=[0,1]).ravel()
        print('tn: ', tn,'fp: ', fp, 'fn: ', fn, 'tp: ', tp)
        #Accuracy for this model
        a = accuracy_score(testing_label[:, index], y_pred)*100
        print("Accuracy : ", a, "%") 
        mean_accuracy += a
        
        #Precision for this model
        b = precision_score(testing_label[:, index], y_pred, average='binary')
        print("Precision : ", b) 
        mean_precision += b
        
        #Recall for this model
        c = recall_score(testing_label[:, index], y_pred, average='binary')
        print("Recall : ", c) 
        mean_recall += c
        
        print("------------------")
        index += 1
        
    
    aa = mean_accuracy/32
    print("Mean Accuracy for all Bits: ", aa, "%")
    
    bb = mean_precision/32
    print("Mean Precision for all Bits: ", bb)
    
    cc = mean_recall/32
    print("Mean Recall for all Bits: ", cc)
    
    
start_time = time.time()
main()
end_time = time.time() - start_time
#print("Running time of program", time.time()-start_time, 'seconds.')
print('Running time of program:  {0:0.2f} seconds'.format(end_time), '({0:0.2f} minutes)'.format(end_time/60))
     
