#Демонстрація ефективності класифікатора побудованого на дереві рішень на основі бази даних ірисів.

from sklearn.datasets import load_iris
iris = load_iris()
#print(iris)
X = iris['data']
Y = iris['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
print(X_train)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))