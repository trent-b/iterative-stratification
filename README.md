# iterative-stratification
iterative-stratification is a project that provides scikit-learn compatible cross validators with stratification for multilabel data.

Presently scikit-learn 0.19.0 provides several cross validators with stratification. However, these cross validators do not offer the ability to stratify _multilabel_ data. This iterative-stratification project offers implementations of MultilabelStratifiedKFold, MultilabelRepeatedStratifiedKFold, and MultilabelStratifiedShuffleSplit with a base algorithm for approximately stratifying multilabel data described in the following paper:

Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data. In: Gunopulos D., Hofmann T., Malerba D., Vazirgiannis M. (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2011. Lecture Notes in Computer Science, vol 6913. Springer, Berlin, Heidelberg.

# Installation
iterative-stratification is tested to work under Python 3.5. The dependency requirements are based on the last scikit-learn release:

scipy(>=0.13.3)
numpy(>=1.8.2)
scikit-learn(>=0.19.0)

iterative-stratification is currently available on the PyPi's repository and you can
install it via pip:
pip install -U iterative-stratification

The package is release also in Anaconda Cloud platform:
conda install -c conda-forge iterative-stratification

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies:
git clone https://github.com/trent-b/iterative-stratification.git
cd iterative-stratification
pip install .

Or install using pip and GitHub:
pip install -U git+https://github.com/trent-b/iterative-stratification.git