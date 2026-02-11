import logging
from typing import Callable

import lightning as L
from omegaconf import DictConfig
from omegaconf.errors import ConfigKeyError
from torch.utils.data import DataLoader

from ..exceptions import (
    EmptyContextError,
    InvalidConfigError,
    MissingDependencyError,
    SetupNotCalledError,
)
from ..utils.instantiators import instantiate_with_context
from ..types import DataSplitIndexedDict, DataSplitIndexedTuple

logger = logging.getLogger(__name__)


class LitGenericDataModule(L.LightningDataModule):

    data_iter_factory: Callable | dict[str, Callable]
    loader_factory: Callable | dict[str, Callable]

    def __init__(
        self,
        data_iter_factory: Callable | dict[str, Callable],
        loader_factory: Callable | dict[str, Callable],
        transform: DictConfig | Callable | None = None,
        collate: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Initialize context
        self.context = {}

        self.data_iter_factory = data_iter_factory
        self.loader_factory = loader_factory

        self.transform = transform

        if collate is not None:
            raise NotImplementedError("Collate function not implemented.")

        self.iterators = DataSplitIndexedDict(train=None, val=None, test=None)
        self._setup_called = False

    @property
    def transform(self) -> DictConfig | Callable | None:
        try:
            value = instantiate_with_context(self._transform, self.context)
            self._transform = value
        except (EmptyContextError, MissingDependencyError):
            # More Context needed: defer instantiation by keeping the transform as DictConfig
            value = self._transform
        except InvalidConfigError:
            # Transform already instantiated into a callable
            value = self._transform
        except AttributeError:
            value = None

        return value

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    @property
    def loaders(self) -> DataSplitIndexedTuple:
        # Multiple loader factories per dataset split
        try:
            loaders = DataSplitIndexedTuple(
                train=self.loader_factory.get("train", None),
                val=self.loader_factory.get("val", None),
                test=self.loader_factory.get("test", None),
            )
            if not set(self.loader_factory.keys()).issubset({"train", "val", "test"}):
                raise InvalidConfigError(
                    "If specifying multiple loader factories, keys must be either "
                    "'train', 'val', 'test'. No other keys are allowed."
                )
        # Single loader factory
        except (AttributeError, ConfigKeyError):
            loaders = DataSplitIndexedTuple(
                train=self.loader_factory,
                val=self.loader_factory,
                test=self.loader_factory,
            )
        return loaders

    @property
    def data_factories(self) -> DataSplitIndexedTuple:
        # Multiple data iterator factories per dataset split
        try:
            factories = DataSplitIndexedTuple(
                train=self.data_iter_factory.get("train", None),
                val=self.data_iter_factory.get("val", None),
                test=self.data_iter_factory.get("test", None),
            )
        # Single data iterator factory
        except (AttributeError, ConfigKeyError):
            factories = DataSplitIndexedTuple(
                train=self.data_iter_factory,
                val=self.data_iter_factory,
                test=self.data_iter_factory,
            )
        return factories

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                # Fit stage requires train and val datasets and updates the context
                self.iterators.train = self.data_factories.train()
                self.iterators.val = self.data_factories.val()

                # Update context with train context
                try:
                    new_context = self.iterators.train.context
                except AttributeError:
                    new_context = {}
                self.context.update(new_context)

                if (transform := self.transform) is not None:
                    try:
                        self.iterators.train.set_transform(transform)
                        self.iterators.val.set_transform(transform)
                    except AttributeError:
                        self.iterators.train.transform = transform
                        self.iterators.val.transform = transform
            case "validate":
                self.iterators.val = self.data_factories.val()
                if (transform := self.transform) is not None:
                    try:
                        self.iterators.val.set_transform(transform)
                    except AttributeError:
                        self.iterators.val.transform = transform
            case "test":
                self.iterators.test = self.data_factories.test()
                if (transform := self.transform) is not None:
                    self.iterators.test.transform = transform
            case "predict":
                raise NotImplementedError("Predict stage not implemented.")
            case _:
                raise ValueError(f"Invalid stage: {stage}")

        self._setup_called = True

    def _validate_setup(self) -> None:
        if not self._setup_called:
            raise SetupNotCalledError()

    def teardown(self, stage: str) -> None:
        match stage:
            case "fit":
                del self.iterators.train
                del self.iterators.val
                self.iterators.train = self.iterators.val = None
            case "validate":
                del self.iterators.val
                self.iterators.val = None
            case "test":
                del self.iterators.test
                self.iterators.test = None
            case "predict":
                raise NotImplementedError("Predict stage not implemented.")
            case _:
                pass

    def train_dataloader(self) -> DataLoader:
        self._validate_setup()
        try:
            # Specify shuffle=True if the factory hasn't already
            loader = self.loaders.train(self.iterators.train, shuffle=True)
        except (TypeError, SyntaxError, ValueError):
            # Factory already has a shuffle argument specified
            loader = self.loaders.train(self.iterators.train)
        return loader

    def val_dataloader(self) -> DataLoader:
        self._validate_setup()
        try:
            # Specify shuffle=False if the factory hasn't already
            loader = self.loaders.val(self.iterators.val, shuffle=False)
        except (TypeError, SyntaxError, ValueError):
            # Factory already has a shuffle argument specified
            loader = self.loaders.val(self.iterators.val)
        return loader

    def test_dataloader(self) -> DataLoader:
        self._validate_setup()
        try:
            # Specify shuffle=False if the factory hasn't already
            loader = self.loaders.test(self.iterators.test, shuffle=False)
        except (TypeError, SyntaxError, ValueError):
            # Factory already has a shuffle argument specified
            loader = self.loaders.test(self.iterators.test)
        return loader
