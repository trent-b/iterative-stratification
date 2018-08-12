"""Package that provides scikit-learn compatible cross validators with stratification for multilabel data"""

from setuptools import find_packages, setup

DISTNAME = 'iterative-stratification'
DESCRIPTION = 'Package that provides scikit-learn compatible cross validators ' + \
              'with stratification for multilabel data'
LONG_DESCRIPTION = 'This iterative-stratification project offers ' + \
                   'implementations of MultilabelStratifiedKFold, ' + \
                   'MultilabelRepeatedStratifiedKFold, and MultilabelStratifiedShuffleSplit with ' + \
                   'a base algorithm for stratifying multilabel data described in the following ' + \
                   'paper: Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification ' + \
                   'of Multi-Label Data. In: Gunopulos D., Hofmann T., Malerba D., Vazirgiannis ' + \
                   'M. (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD ' + \
                   '2011. Lecture Notes in Computer Science, vol 6913. Springer, Berlin, ' + \
                   'Heidelberg.'
MAINTAINER = 'Trent J. Bradberry'
MAINTAINER_EMAIL = 'trentjason@hotmail.com'
URL = 'https://github.com/trent-b/iterative-stratification'
LICENSE = 'BSD 3'
DOWNLOAD_URL = 'https://github.com/trent-b/iterative-stratification'
VERSION = '0.1.6'
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering']

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES)
