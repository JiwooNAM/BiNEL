import warnings


class InvalidNameFormatError(ValueError):
    """Raised when the name format is invalid."""


class MissingRequiredParameterError(ValueError):
    """Raised when a required parameter is missing."""


class InvalidModalityError(ValueError):
    """Raised when an invalid modality is specified."""


def get_datapath(
    path,
    split,
    embedding_file_name,
    embedding_key,
    *,
    name: str | None = None,
    omics_embedding: str | None = None,
    imaging_embedding: str | None = None,
    imaging_revision: str | None = None,
    omics_revision: str | None = None,
    include_seen: bool | None = None,
    include_unseen: bool | None = None,
    batch_select: list[str] | list[int] | None = None,
    replicate_select: list[str] | list[int] | None = None,
    treatment_select: list[str] | None = None,
    name_separator: str = "//",
):
    if name:
        if omics_embedding or imaging_embedding or omics_revision or imaging_revision:
            raise ValueError(
                "If name is provided, omics_embedding, imaging_embedding, "
                "omics_revision, and imaging_revision must be None"
            )

        # Parse the name into its components
        match name.split(name_separator):
            case [modality, embedding, *revision]:
                # Slash-separated format: modality<name_separator>embedding<name_separator>revision
                if modality == "multimodal":
                    raise InvalidNameFormatError(
                        "For multimodal models, please use the parameter-based format "
                        "with separate embedding and revision parameters for each "
                        "modality"
                    )
                # For unimodal models, use the same embedding and revision
                omics_embedding = imaging_embedding = embedding
                omics_revision = imaging_revision = revision or ""
            case _:
                raise InvalidNameFormatError(
                    f"Name format must be "
                    f"'modality{name_separator}embedding{name_separator}revision'."
                )
    else:
        if omics_embedding and imaging_embedding:
            modality = "multimodal"
        elif omics_embedding:
            modality = "rnaseq"
        elif imaging_embedding:
            modality = "imaging"
        else:
            raise MissingRequiredParameterError(
                "Either name or any of the following parameters must be provided: "
                "omics_embedding, imaging_embedding."
            )
        # Handle optional revisions
        omics_revision = omics_revision or ""
        imaging_revision = imaging_revision or ""

    if modality == "multimodal" or modality == "rnaseq":
        # Parse revision into path and extras
        revision_path, *omics_extras = omics_revision.split(name_separator)
        omics_data_path = (
            f"{path}/features/rnaseq/{omics_embedding}/{revision_path}/"
            f"{embedding_file_name}"
        )
    else:
        omics_data_path = omics_extras = None

    if modality == "multimodal" or modality == "imaging":
        # Parse revision into path and extras
        revision_path, *imaging_extras = imaging_revision.split(name_separator)
        imaging_data_path = (
            f"{path}/features/imaging/{imaging_embedding}/{revision_path}/"
            f"{embedding_file_name}"
        )
    else:
        imaging_data_path = imaging_extras = None

    paths = {
        "omics": omics_data_path,
        "imaging": imaging_data_path,
    }

    match modality:
        case "multimodal":
            warnings.warn("Joint loading of multimodal data is not yet supported.")
            path = paths
            kwargs = dict(
                include_seen=include_seen,
                include_unseen=include_unseen,
                batch_select=batch_select,
                replicate_select=replicate_select,
                treatment_select=treatment_select,
            )
        case "rnaseq":
            if omics_extras:
                raise NotImplementedError(
                    "Loading of omics data with extras is not yet supported."
                )
            path = paths["omics"]
            data_src = f"obsm:{embedding_key}"
            kwargs = dict(
                data_src=data_src,
                include_seen=include_seen,
                include_unseen=include_unseen,
                batch_select=batch_select,
                replicate_select=replicate_select,
                treatment_select=treatment_select,
            )
        case "imaging":
            path = paths["imaging"]
            kwargs = dict(
                data_src=imaging_extras[0] if imaging_extras else "X",
                include_seen=include_seen,
                include_unseen=include_unseen,
                batch_select=batch_select,
                replicate_select=replicate_select,
                treatment_select=treatment_select,
            )
        case _:
            raise ValueError(f"Invalid dataset name: {name}")

    return path, kwargs
