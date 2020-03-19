from seaborn import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris_data_set = load_dataset("iris")

iris_class = iris_data_set.pop("species")
iris_features = iris_data_set

features_train, features_test, class_train, class_test = train_test_split(iris_features, iris_class, test_size=0.3)

decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(features_train, class_train)

predictions = decision_tree_classifier.predict(features_test)

print(accuracy_score(class_test, predictions))
print(confusion_matrix(class_test, predictions))
print(classification_report(class_test, predictions))
