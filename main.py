from sklearn import datasets

#for traing our datasets -1
from sklearn.model_selection import train_test_split

#for which section is match to our sample -2
from sklearn.neighbors import KNeighborsClassifier

#for accuracy score -3
from sklearn.metrics import accuracy_score


#load data form "datasets"
data_set = datasets.load_iris()


# load data from data_sets
features = data_set.data
label = data_set.target

#Section no 2 | sepal lenth, sepal width, petal length, petal width
features_train,features_test,label_train,label_test = train_test_split(features,label,test_size=.5)
#print(len(features_train))

#call Kneighbors and call "fit" for traing the data | Section 3
my_classifier = KNeighborsClassifier()
my_classifier.fit(features_train,label_train)
#call the "predit" to test the features from "myclassifier"
prediction = my_classifier.predict(features_test)

#print(prediction)

#section no 4
print("Score is:",accuracy_score(label_test,prediction))

#versicolor
iris1 = [[4.7,2.5,3.1,1.2]]

#Extra: Taking 1D array as input
#------------
#print("Enter Values:")
#iris1 = list(map(float,input().split()))
#print(iris1)
#------------

# TO print the flower name using "KNeigbbour"
iris_predic = my_classifier.predict(iris1)

if iris_predic[0]==0:
    print("Flower: Setosa")

if iris_predic[0]==1:
    print("Flower: Versicolor")

if iris_predic[0]==2:
    print("Flower: Virginica")