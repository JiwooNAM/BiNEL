# Copyright (c) 2024 Altos Labs, Inc. Altos Confidential.

import itertools
import functools
from pathlib import Path
from typing import Collection, Sequence

import torch
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer

from .core import Transform
from ...types import ExampleMultiLabel, BatchMultiLabel


class OneHotEncode(Transform):
    """One-hot encode a categorical variable.

    Attributes:
        onehot_encoder: the wrapped encoder instance
    """

    one_hot_encoder: OneHotEncoder

    def __init__(
        self,
        categories: Collection[str] | None = None,
        path_to_categories: Path | None = None,
        **kwargs,
    ):
        if path_to_categories and categories is None:
            categories = [np.load(Path(path_to_categories), allow_pickle=True)]
        elif categories is not None and not path_to_categories:
            categories = [list(categories)]
        else:
            raise ValueError(
                "Either categories or path_to_categories must be provided but not both."
            )

        self.one_hot_encoder = OneHotEncoder(
            categories=categories,
            sparse_output=False,
            **kwargs,
        )

    def __call__(self, labels: Sequence[str]):
        string_array = np.array(labels).reshape(-1, 1)
        encoded = self.one_hot_encoder.fit_transform(string_array)
        return torch.tensor(encoded)

    def __repr__(self):
        _base = super().__repr__()
        categories = ", ".join(self.one_hot_encoder.categories[0])
        return _base.format(categories)


class LabelEncode(Transform):
    """Label encode categorical variables.

    Attributes:
        ordinal_encoder: sklearn.preprocessing.OrdinalEncoder
    """

    ordinal_encoder: OrdinalEncoder

    def __init__(
        self,
        categories: Sequence[str] | None = None,
        path_to_categories: Path | None = None,
    ):
        if path_to_categories and categories is None:
            categories = [np.load(Path(path_to_categories), allow_pickle=True)]
        elif categories is not None and not path_to_categories:
            categories = [np.array(categories)]
        else:
            raise ValueError(
                "Either categories or path_to_categories must be provided but not both."
            )
        self.ordinal_encoder = OrdinalEncoder(categories=categories)

    def __call__(self, labels: Sequence[str]) -> torch.Tensor:
        string_array = np.array(labels).reshape(-1, 1)
        return torch.tensor(
            self.ordinal_encoder.fit_transform(string_array), dtype=torch.int64
        )

    def __repr__(self):
        _base = super().__repr__()
        categories = ", ".join(self.ordinal_encoder.categories[0])
        return _base.format(categories)


class MultiLabelEncode(Transform):
    """Transforms a sequence of labels into a binary vector.

    Attributes:
        label_binarizer: the wrapped binarizer instance

    Raises:
        ValueError: if any of the labels are not found in the encoder classes
    """

    label_binarizer: MultiLabelBinarizer

    def __init__(self, classes: Collection[str]):
        self.label_binarizer = MultiLabelBinarizer(
            classes=list(classes), sparse_output=False
        )

    @functools.cached_property
    def classes(self):
        return set(self.label_binarizer.classes)

    def __call__(self, labels: ExampleMultiLabel | BatchMultiLabel) -> torch.Tensor:
        # If labels is a single example, convert it to a batch
        if not labels or isinstance(labels[0], str):
            labels = [labels]
        self._check_inputs(labels)
        encoded = self.label_binarizer.fit_transform(labels)
        return torch.from_numpy(encoded)

    def _check_inputs(self, labels: BatchMultiLabel):
        unique_labels = set(itertools.chain.from_iterable(labels))
        if not unique_labels <= self.classes:
            missing_labels = unique_labels - self.classes
            raise ValueError(
                f"Labels {missing_labels} not found in the encoder classes {self.classes}"
            )

    def __repr__(self):
        _base = super().__repr__()
        classes = ", ".join(self.label_binarizer.classes)
        return _base.format(classes)
