import numpy 
import pandas 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
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
    clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
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
    
 








#importing data
'''
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
'''

'''
filename = 'training_data'
with open(filename) as f:
    newText=f.read().replace('\t', '')
with open(filename, "w") as f:
    f.write(newText)
with open(filename) as f:
    newText=f.read()
    myString = ",".join(newText)
    myString = myString.replace(',\n,', '\n')
    myString = myString[:-2]
with open(filename, "w") as f:
    f.write(myString)

raw_data0 = open(filename, 'rt')
training_data = numpy.loadtxt(raw_data0, delimiter=",")
print(training_data.shape)




filename = 'training_label'
with open(filename) as f:
    newText=f.read()
    myString = ",".join(newText)
    myString = myString.replace(',\n,', '\n')
    myString = myString[:-2]
with open(filename, "w") as f:
    f.write(myString)

raw_data1 = open(filename, 'rt')
training_label = numpy.loadtxt(raw_data1, delimiter=",")
print(training_label.shape)




filename = 'testing_data'
with open(filename) as f:
    newText=f.read().replace('\t', '')
with open(filename, "w") as f:
    f.write(newText)
with open(filename) as f:
    newText=f.read()
    myString = ",".join(newText)
    myString = myString.replace(',\n,', '\n')
    myString = myString[:-2]
with open(filename, "w") as f:
    f.write(myString)

raw_data2 = open(filename, 'rt')
testing_data = numpy.loadtxt(raw_data2, delimiter=",")
print(testing_data.shape)





filename = 'testing_label'
with open(filename) as f:
    newText=f.read()
    myString = ",".join(newText)
    myString = myString.replace(',\n,', '\n')
    myString = myString[:-2]
with open(filename, "w") as f:
    f.write(myString)

raw_data3 = open(filename, 'rt')
testing_label = numpy.loadtxt(raw_data3, delimiter=",")
print(testing_label.shape)





list =[]
#input1=13
#input2=29
#label=29
input1 = 0
input2 = input1 +16
label=input2

while input1<16:
    print("TRIAL " + str(input1))
    print(input1)
    print(input2)
    print(label)
    
    print(training_data[:, [input1, input2]])
    print(training_label[:, label])
    clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
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
        #print("Report : ", 
        #classification_report(testing_label[:, index2], y_pred)) 
        
        index = index+1
        index2 +=1
    ab = total_mean_accuracy/32
    list.append(ab)
    print("total mean accuracy:" + str(ab))

    input1 += 1
    input2 +=1
    label=input2

print(list)
print(max(list))
'''
'''
print("predictions for bit4s")
y_pred = clf.predict(testing_data[:, [13, 29]]) 
print(testing_data[:, [13, 29]])
print(training_label[:, 29])
print("Predicted values:") 
print(y_pred) 

print("Confusion Matrix: ", 
confusion_matrix(training_label[:, 29], y_pred)) 
    
print ("Accuracy : ", 
accuracy_score(training_label[:, 29],y_pred)*100) 
    
print("Report : ", 
classification_report(training_label[:, 29], y_pred)) 





print("predictions for bit0s")
y_pred = clf.predict(testing_data[:, [15, 31]]) 
print(testing_data[:, [15, 31]])
print(training_label[:, 31])
print("Predicted values:") 
print(y_pred) 

print("Confusion Matrix: ", 
confusion_matrix(training_label[:, 31], y_pred)) 
    
print ("Accuracy : ", 
accuracy_score(training_label[:, 31],y_pred)*100) 
    
print("Report : ", 
classification_report(training_label[:, 31], y_pred)) 

'''




#clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(training_data, training_label)
#clf.predict(training_data[:2, :])
#clf.predict_proba(training_data[:2, :]) 
#clf.score(training_data, training_label)




'''
#str1 = " " 
#str1.join(result)
with open(filename, "w") as f:
    f.write(str1)

training_data= numpy.genfromtxt('training_data', delimiter='\n', dtype=numpy.bool_, names=('column1'))
print(training_data)
'''

'''
filename = 'training_data'
raw_data0 = open(filename, 'rt')
training_data = numpy.loadtxt(raw_data0, delimiter="\t")
print(training_data.shape)
charar = numpy.chararray(training_data)
print(charar)
'''




'''
#importing data

filename = 'training_data'
raw_data0 = open(filename, 'rt')
training_data = numpy.loadtxt(raw_data0, delimiter="\n")
print(training_data.shape)


filename = 'training_label'
raw_data1 = open(filename, 'rt')
training_label = numpy.loadtxt(raw_data1, delimiter="\n")
print(training_label.shape)


filename = 'testing_data'
raw_data2 = open(filename, 'rt')
testing_data = numpy.loadtxt(raw_data2, delimiter="\n")
print(testing_data.shape)

filename = 'testing_label'
raw_data3 = open(filename, 'rt')
testing_label = numpy.loadtxt(raw_data3, delimiter="\n")
print(testing_label.shape)


clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :]) 
clf.score(X, y)






filename = 'training_data'
with open(filename) as f:
    newText=f.read().replace('\t', '')
with open(filename, "w") as f:
    f.write(newText)

filename = 'training_data'
with open(filename) as f:
    result = [list(line.rstrip()) for line in f]
#print(result)

training_data = numpy.asarray(result)
print(training_data.shape)


filename = 'training_label'
with open(filename) as f:
    result2 = [list(line.rstrip()) for line in f]
#print(result)

training_label = numpy.asarray(result2)
print(training_label.shape)




'''