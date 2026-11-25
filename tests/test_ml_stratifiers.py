
from unittest import TestCase, main

import numpy as np


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class TestMultilabelStratifiedKFold(TestCase):

    def test_seeded_tie_breaking_without_shuffle(self):
        y = np.repeat([[0, 0], [0, 1], [1, 1], [1, 0]], 2, axis=0)
        x = np.zeros(len(y))
        for seed, expected in [(0, [[1, 2, 5, 7], [0, 3, 4, 6]]),
                               (1, [[1, 3, 4, 6], [0, 2, 5, 7]])]:
            splitter = MultilabelStratifiedKFold(n_splits=2, random_state=seed)
            for _ in range(2):
                folds = [test.tolist() for _, test in splitter.split(x, y)]
                self.assertEqual(folds, expected)

    def test_integration(self):

        def _test(n_labels, n_samples, n_splits):
            x = np.zeros((n_samples, 2))  # This is not used in the split
            y = np.random.randint(0, 2, size=(n_samples, n_labels))
            mskf = MultilabelStratifiedKFold(n_splits=n_splits)

            for train_index, test_index in mskf.split(x, y):
                self.assertEqual(len(set(train_index) & set(test_index)), 0)
                y_train = np.sum(y[train_index], axis=0)
                y_train = y_train / np.sum(y_train)
                y_test = np.sum(y[test_index], axis=0)
                y_test = y_test / np.sum(y_test)
                for i, (r1, r2) in enumerate(zip(y_train, y_test)):
                    self.assertAlmostEqual(r1, r2, delta=0.07,
                                           msg="n_labels={}, n_samples={}, n_splits={}\n"
                                           .format(n_labels, n_samples, n_splits) +
                                               "i={}: {} vs {}\n".format(i, r1, r2))
        np.random.seed(12345)
        _test(3, 500, n_splits=7)
        _test(5, 500, n_splits=7)
        _test(10, 500, n_splits=7)


if __name__ == "__main__":
    main()
