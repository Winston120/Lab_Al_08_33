#Ще один приклад використання бази даних ірисів для побудови
#класифікатора.
import sklearn as sk
from sklearn import datasets

iris = sk.datasets.load_iris()
X_iris, Y_iris = iris.data, iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_iris, Y_iris, test_size=0.30,random_state=33)

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)

from sklearn import metrics

print (metrics.classification_report(y_test, y_pred, target_names=iris.target_names))