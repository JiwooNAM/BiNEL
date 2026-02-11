from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from collections.abc import Callable

import pandas as pd
import anndata as ad
import torch

from ..base import DataStore
from ..mixin import ContextMixin, ObservationGroupingMixin
from ....types import Spec


T = TypeVar("T")


@dataclass
class Samples(DataStore, ContextMixin, ObservationGroupingMixin):
    """A dataset of samples where each sample represents an experiment.

    Each sample has numerical data and observations associated with it. It uses
    the DataStore class as a backend to store the data and observations.

    It provides a collection of factory methods to read data from different
    sources such as AnnData objects, CSV files, etc. and create a dataset.

    Factories:
        from_anndata: create a dataset from an AnnData object.
        from_csv: create a dataset from a CSV file.
        auto_factory: automatically determine the data format and create the dataset.
    """

    def __repr__(self) -> str:
        return DataStore.__repr__(self)

    @classmethod
    def from_anndata(
        cls,
        file_path: str,
        data_src: str,
        filters: dict[str, list] | None = None,
        obs_keys: list[str] | None = None,
        context_params: list[str] | None = None,
        compute_context: Callable | None = None,
        **kwargs,
    ):
        """Create a UniModalSingleCell dataset from an AnnData object.

        Args:
            file_path: path to the h5ad file.
            data_src: the data source spec as a <field:keys> schema that
              specifies the data source in the AnnData object. For example,
              "X" specifies the expression matrix, "obsm:scvi" specifies the
              scvi latent space embeddings.
            obs_keys: the keys to use from the observations.
        """

        # First read only obs to filter indices
        adata = ad.read_h5ad(file_path, backed="r")
        obs = adata.obs.copy()

        # Get indices that pass all filters
        if filters:
            mask = pd.Series(True, index=obs.index)
            for key, values in filters.items():
                if key in obs:
                    mask &= obs[key].isin(values)
                else:
                    raise KeyError(f"Filter key '{key}' not found in adata.obs")

            # Get indices that pass all filters
            valid_indices = obs[mask].index

            if len(valid_indices) == 0:
                raise ValueError(
                    "No data remains after applying filters. "
                    "Please check your filter criteria."
                )
        else:
            valid_indices = obs.index

        adata = adata[valid_indices, :]

        # Check if any data remains after filtering
        if len(adata) == 0:
            raise ValueError(
                "No data remains after applying filters. "
                "Please check your filter criteria."
            )

        # Get data store
        src, *keys = data_src.split(":")
        data_store = getattr(adata, src)
        # Parse keys
        if len(keys) > 1:
            raise ValueError(
                f"Invalid data source: {data_src}. "
                f"Should be <src>:<key_1, ..., key_N>."
            )
        if keys:
            keys = keys[0].split(",")
            if len(keys) == 1:
                data_store = data_store[keys[0]]
            else:
                # Wrap into a torch.utils.data.StackDataset since the obsm of
                # anndata object is a dictionary of tensors
                data_store = torch.utils.data.StackDataset(
                    *(data_store[key] for key in keys)
                )

        # Get observations
        obs_keys = obs_keys or adata.obs.columns
        obs = adata.obs[obs_keys]

        return cls(
            data_store,
            obs,
            context_params=context_params,
            compute_context=compute_context,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        file_path: str,
        data_src: str,
        include_seen: bool | None = None,
        include_unseen: bool | None = None,
        batch_select: list[str] | list[int] | None = None,
        replicate_select: list[str] | list[int] | None = None,
        treatment_select: list[str] | None = None,
        obs_keys: list[str] | None = None,
        context_params: list[str] | None = None,
        compute_context: Callable | None = None,
        **kwargs,
    ):
        """Create a UniModalSingleCell dataset from an AnnData object.

        Args:
            file_path: path to the h5ad file.
            data_src: the data source spec as a <field:keys> schema that
              specifies the data source in the AnnData object. For example,
              "X" specifies the expression matrix, "obsm:scvi" specifies the
              scvi latent space embeddings.
            obs_keys: the keys to use from the observations.
        """

        # Validate seen/unseen parameters
        if include_seen is None and include_unseen is None:
            raise ValueError(
                "At least one of 'include_seen' or 'include_unseen' must be specified. "
                "Both cannot be None simultaneously."
            )

        # Create a filter based on seen/unseen parameters
        seen_filter = []

        if include_seen and include_unseen:
            # Include both seen and unseen samples
            seen_filter = None
        elif include_seen:
            # Include only seen samples
            seen_filter = [True]
        elif include_unseen:
            # Include only unseen samples
            seen_filter = [False]

        # Convert batch_select to list of strings if it contains integers
        if batch_select is not None and isinstance(batch_select[0], int):
            batch_select = [str(x) for x in batch_select]

        # Convert replicate_select to list of strings if it contains integers 
        if replicate_select is not None and isinstance(replicate_select[0], int):
            replicate_select = [str(x) for x in replicate_select]

        filters = {}

        if seen_filter is not None:
            filters["Seen"] = seen_filter

        if batch_select is not None:
            filters["batch"] = batch_select

        if replicate_select is not None:
            filters["Replicate"] = replicate_select

        if treatment_select is not None:
            filters["Treatment"] = treatment_select

        # If no filters were added, set filters to None
        if not filters:
            filters = None

        return cls.from_anndata(
            file_path,
            data_src,
            filters=filters,
            obs_keys=obs_keys,
            context_params=context_params,
            compute_context=compute_context,
            **kwargs,
        )

    @classmethod
    def auto_factory(
        cls,
        file_path: Path | str | tuple[str, dict[str, Any]],
        context_params=None,
        compute_context=None,
        **kwargs: Spec,
    ):
        """Automatically determine data format to create the dataset."""
        try:
            file_path, extra_kwargs = file_path
        except ValueError:
            file_path = file_path
            extra_kwargs = {}
        file_path = Path(file_path)
        kwargs.update(extra_kwargs)

        extension = file_path.suffix.lstrip(".")

        match extension.lower():
            case "h5ad":
                dataset = cls.from_embeddings(file_path, **kwargs)
            case _:
                raise ValueError(f"Unsupported file format: {extension}")
        return dataset