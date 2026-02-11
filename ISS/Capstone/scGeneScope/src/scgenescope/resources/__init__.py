import pathlib
from enum import StrEnum


class ResourceRegistry(StrEnum):
    """Maps resources name to their file name."""

    seen_treatments = "treatment_categories_aplusv2_seen.npy"
    unseen_treatments = "treatment_categories_aplusv2_unseen.npy"
    all_treatments = "treatment_categories_aplusv2_all.npy"
    all_treatments_no_DMSO = "treatment_categories_aplusv2_all_no_DMSO.npy"

    def get_path(self) -> pathlib.Path:
        """Get the full path to this resource."""
        resource_path = pathlib.Path(__file__).parent.resolve()
        return resource_path / self.value

    @classmethod
    def from_name(cls, name: str) -> "ResourceRegistry":
        """Get a ResourceRegistry member from its name.

        Args:
            name: The name of the resource (e.g. "treatment_categories_seen")

        Returns:
            The corresponding ResourceRegistry member

        Raises:
            ValueError: If no member with the given name exists
        """
        try:
            return cls[name]
        except KeyError:
            raise ValueError(f"'{name}' is not a valid {cls.__name__} member name")


def request_resource(resource: ResourceRegistry | str) -> pathlib.Path:
    """Get the full path to a resource.

    Args:
        resource: A ResourceRegistry member or string name of a member

    Returns:
        Path to the resource file
    """
    if isinstance(resource, str):
        resource = ResourceRegistry.from_name(resource)
    return resource.get_path()
