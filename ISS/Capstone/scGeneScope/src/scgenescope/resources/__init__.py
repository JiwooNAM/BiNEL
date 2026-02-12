import pathlib
from enum import StrEnum


class ResourceRegistry(StrEnum):
    """Maps resources name to their file name."""

    seen_treatments = "treatment_categories_aplusv2_seen.npy"
    unseen_treatments = "treatment_categories_aplusv2_unseen.npy"
    all_treatments = "treatment_categories_aplusv2_all.npy"
    all_treatments_no_DMSO = "treatment_categories_aplusv2_all_no_DMSO.npy"
    phenacetin = "singletreat/Phenacetin.npy"
    pq401 = "singletreat/PQ401.npy"
    splitomicin = "singletreat/Splitomicin.npy"
    r_mg132 = "singletreat/(R)-MG132.npy"
    r_roscovitine = "singletreat/(R)-Roscovitine.npy"
    wy_14643 = "singletreat/Wy_14643_Pirinixic_Acid.npy"
    fluocinonide = "singletreat/Fluocinonide.npy"
    caffeine = "singletreat/Caffeine.npy"
    ly303511 = "singletreat/LY303511_(hydrochloride).npy"
    simvastatin = "singletreat/Simvastatin.npy"
    colchicine = "singletreat/Colchicine.npy"
    pantoprazole = "singletreat/Pantoprazole.npy"
    cycloheximide = "singletreat/Cycloheximide.npy"
    benzbromarone = "singletreat/Benzbromarone.npy"
    thapsigargin = "singletreat/Thapsigargin.npy"
    bay_11 = "singletreat/BAY_11-7082.npy"
    cgk_733 = "singletreat/CGK-733.npy"
    pd_98059 = "singletreat/PD-98059.npy"
    gw_843682x = "singletreat/GW-843682X.npy"
    pma = "singletreat/12-O-tetradecanoylphorbol-13-acetate.npy"
    skii = "singletreat/SKII.npy"
    amg_900 = "singletreat/AMG-900.npy"
    dbeq = "singletreat/DBeQ.npy"
    daporinad = "singletreat/Daporinad_FK-866.npy"
    vorinostat = "singletreat/Vorinostat_SAHA.npy"
    quinidine = "singletreat/Quinidine.npy"
    aloxistatin = "singletreat/Aloxistatin_E-64d.npy"
    harman = "singletreat/HARMAN.npy"

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
