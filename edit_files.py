import numpy 
import pandas 



#importing data

filename = 'testing_data2'
'''
with open(filename) as f:
    newText=f.read().replace('\t', '')
with open(filename, "w") as f:
    f.write(newText)


with open(filename) as f:
    newText=f.read()
    myString = ",".join(newText)
    myString = myString.replace(',\n,', '\n')
    
    

with open(filename, "w") as f:
    f.write(myString)

~~~~~
file = open(filename, "r+")
for line in file:
    print(line)
    myString = ",".join(line)
    file.write(myString)
    
file.close()

'''


raw_data0 = open(filename, 'rt')
training_data = numpy.loadtxt(raw_data0, delimiter=",")
print(training_data.shape)
#training_data.head(4)
