# iterative-stratification
iterative-stratification is a project that provides scikit-learn compatible cross validators with stratification for multilabel data.

Presently scikit-learn 0.19.0 provides several cross validators with stratification. However, these cross validators do not offer the ability to stratify multilabel data. This iterative-stratification project offers implementations of MultilabelStratifiedKFold, MultilabelRepeatedStratifiedKFold, and MultilabelStratifiedShuffleSplit with a base algorithm for approximately stratifying multilabel data described in the following paper:

Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data. In: Gunopulos D., Hofmann T., Malerba D., Vazirgiannis M. (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2011. Lecture Notes in Computer Science, vol 6913. Springer, Berlin, Heidelberg.
