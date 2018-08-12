"""Package that provides scikit-learn compatible cross validators with stratification for multilabel data.
``iterative-stratification`` offers implementations of MultilabelStratifiedKFold,
MultilabelRepeatedStratifiedKFold, and MultilabelStratifiedShuffleSplit with a base algorithm for stratifying
multilabel
Subpackages
-----------
ml_stratifiers
    Module that implements MultilabelStratifiedKFold,
    MultilabelRepeatedStratifiedKFold, and MultilabelStratifiedShuffleSplit.
"""

__version__ = '0.1.6'

# list all submodules available in iterstrat and version
__all__ = ['ml_stratifiers', '__version__']
