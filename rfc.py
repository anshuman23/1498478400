import arff
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

arffData  = arff.load('NIMS.arff')
data = []
for row in arffData:
    data.append(list(row))

data = np.array(data)

X = data[:,:22]
y = data[:,22]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Xtrain,Xtest, ytrain,ytest  = train_test_split(X_scaled,y)

RFC = RandomForestClassifier(n_estimators = 5, verbose = True)

clf = RFC.fit(Xtrain,ytrain)

acc_train = accuracy_score(clf.predict(Xtrain),ytrain)
acc_test = accuracy_score(clf.predict(Xtest), ytest)

print "Accuracy on training set is : " + str(acc_train) + "\nAccuracy on testing set is : " + str(acc_test)
