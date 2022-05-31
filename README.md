
[![Build Status](https://travis-ci.org/vfdev-5/iterative-stratification.svg?branch=master)](https://travis-ci.org/vfdev-5/iterative-stratification)
[![Coverage Status](https://coveralls.io/repos/github/vfdev-5/iterative-stratification/badge.svg?branch=master)](https://coveralls.io/github/vfdev-5/iterative-stratification?branch=master)

# iterative-stratification
iterative-stratification is a project that provides [scikit-learn](http://scikit-learn.org/) compatible cross validators with stratification for multilabel data.

Presently scikit-learn provides several cross validators with stratification. However, these cross validators do not offer the ability to stratify _multilabel_ data. This iterative-stratification project offers implementations of MultilabelStratifiedKFold, MultilabelRepeatedStratifiedKFold, and MultilabelStratifiedShuffleSplit with a base algorithm for stratifying multilabel data described in the following paper:

Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data. In: Gunopulos D., Hofmann T., Malerba D., Vazirgiannis M. (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2011. Lecture Notes in Computer Science, vol 6913. Springer, Berlin, Heidelberg.

## Requirements
iterative-stratification has been tested under Python 3.4 through 3.8 with the following dependencies:
- scipy(>=0.13.3)
- numpy(>=1.8.2)
- scikit-learn(>=0.19.0)

## Installation
iterative-stratification is currently available on the PyPi repository and can be installed via pip:
```
pip install iterative-stratification
```

## Toy Examples
The multilabel cross validators that this package provides may be used with the scikit-learn API in the same manner as any other cross validators. For example, these cross validators may be passed to [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) or [cross_val_predict](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html). Below are some toy examples of the direct use of the multilabel cross validators.

### MultilabelStratifiedKFold
```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np

X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])

mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=0)

for train_index, test_index in mskf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
```
Output:
```
TRAIN: [0 3 4 6] TEST: [1 2 5 7]
TRAIN: [1 2 5 7] TEST: [0 3 4 6]
```
### RepeatedMultilabelStratifiedKFold
```python
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
import numpy as np

X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])

rmskf = RepeatedMultilabelStratifiedKFold(n_splits=2, n_repeats=2, random_state=0)

for train_index, test_index in rmskf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
```
Output:
```
TRAIN: [0 3 4 6] TEST: [1 2 5 7]
TRAIN: [1 2 5 7] TEST: [0 3 4 6]
TRAIN: [0 1 4 5] TEST: [2 3 6 7]
TRAIN: [2 3 6 7] TEST: [0 1 4 5]
```
### MultilabelStratifiedShuffleSplit
```python
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])

msss = MultilabelStratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)

for train_index, test_index in msss.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
```
Output:
```
TRAIN: [1 2 5 7] TEST: [0 3 4 6]
TRAIN: [2 3 6 7] TEST: [0 1 4 5]
TRAIN: [1 2 5 6] TEST: [0 3 4 7]
```
