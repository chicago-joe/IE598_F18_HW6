# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

test_scores = []

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
test_scores.append(tree.score(X_test, y_test))
print("Sample 1 Accuracy Score: ", tree.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2, stratify=y)
tree.fit(X_train, y_train)
test_scores.append(tree.score(X_test, y_test))
print("Sample 2 Accuracy Score: ", tree.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3, stratify=y)
tree.fit(X_train, y_train)
test_scores.append(tree.score(X_test, y_test))
print("Sample 3 Accuracy Score: ", tree.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4, stratify=y)
tree.fit(X_train, y_train)
test_scores.append(tree.score(X_test, y_test))
print("Sample 4 Accuracy Score: ", tree.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5, stratify=y)
tree.fit(X_train, y_train)
test_scores.append(tree.score(X_test, y_test))
print("Sample 5 Accuracy Score: ", tree.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=6, stratify=y)
tree.fit(X_train, y_train)
test_scores.append(tree.score(X_test, y_test))
print("Sample 6 Accuracy Score: ", tree.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7, stratify=y)
tree.fit(X_train, y_train)
test_scores.append(tree.score(X_test, y_test))
print("Sample 7 Accuracy Score: ", tree.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7, stratify=y)
tree.fit(X_train, y_train)
test_scores.append(tree.score(X_test, y_test))
print("Sample 8 Accuracy Score: ", tree.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=9, stratify=y)
tree.fit(X_train, y_train)
test_scores.append(tree.score(X_test, y_test))
print("Sample 9 Accuracy Score: ", tree.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10, stratify=y)
tree.fit(X_train, y_train)
test_scores.append(tree.score(X_test, y_test))
print("Sample 10 Accuracy Score: ", tree.score(X_test, y_test))

print(" ")
print('Training Set Accuracy Score: %.3f' % tree.score(X=X_train, y=y_train))
print("Test Set Accuracy Scores: ", test_scores)
print('Mean Score (Test): %.3f' % np.mean(test_scores))
print('Std. Deviation (Test): %.3f' % np.std(test_scores))
print(" ")

# Cross Validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
cv_scores = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=10, n_jobs=1)
tree.fit(X_train, y_train)

print("CV Accuracy Scores (folds): ", cv_scores)
print('CV Mean Score: %.3f' % np.mean(cv_scores))
print('CV Std. Deviation: %.3f' % np.std(cv_scores))
print('CV Accuracy Score: %.3f +/- %.3f' % (np.mean(cv_scores), np.std(cv_scores)))
y_test_pred = tree.predict(X_test)
print('Test Set Accuracy Score: %.3f' % accuracy_score(y_test, y_test_pred))



#################################################################################################
print(" ")
print("#########################################################################################")
print('My name is Joseph Loss')
print('My NetID is loss2')
print('I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.')