#Для реалізації методу класифікації kNN будемо
#використовувати ту ж саму базу даних квіток ірисів, яка міститься у
#бібліотеці scikit-learn. Окрім цього виведемо у консоль важливі
#значення ефективності запропонованого алгоритму.
import sklearn as sk
from sklearn import datasets

iris = sk.datasets.load_iris()
X_iris, Y_iris = iris.data, iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_iris, Y_iris, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))