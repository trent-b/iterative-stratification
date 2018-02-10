# iterative-stratification
iterative-stratification is a project that provides [scikit-learn](http://scikit-learn.org/) compatible cross validators with stratification for multilabel data.

Presently scikit-learn 0.19.0 provides several cross validators with stratification. However, these cross validators do not offer the ability to stratify _multilabel_ data. This iterative-stratification project offers implementations of MultilabelStratifiedKFold, MultilabelRepeatedStratifiedKFold, and MultilabelStratifiedShuffleSplit with a base algorithm for approximately stratifying multilabel data described in the following paper:

Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data. In: Gunopulos D., Hofmann T., Malerba D., Vazirgiannis M. (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2011. Lecture Notes in Computer Science, vol 6913. Springer, Berlin, Heidelberg.

## Requirements
iterative-stratification is tested to work under Python 3.5 with the following dependencies:
- scipy(>=0.13.3)
- numpy(>=1.8.2)
- scikit-learn(>=0.19.0)

## Installation
iterative-stratification is currently available on the PyPi repository and can be installed via pip:
```
pip install -U iterative-stratification
```
\
The package is also released on the Anaconda Cloud platform:
```
conda install -c conda-forge iterative-stratification
```
