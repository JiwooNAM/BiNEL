"""Custom exceptions for the scgenescope package."""


class InvalidConfigError(TypeError):
    """Raised when the config is invalid."""


class ContextError(TypeError):
    """Raised when there is an error in instantiating a class with dependencies."""


class EmptyContextError(ContextError):
    """Raised when the context is empty but dependencies are requested."""


class MissingDependencyError(ContextError):
    """Raised when a dependency is missing from the context."""


class SetupNotCalledError(RuntimeError):
    """Raised when trying to access dataloaders before calling setup().

    This error occurs when attempting to use train_dataloader(), val_dataloader(),
    or test_dataloader() methods before the setup() method has been called with
    the appropriate stage.
    """

    def __init__(self, message: str | None = None) -> None:
        if message is None:
            message = (
                "setup() must be called before accessing dataloaders. "
                "Please call setup() with the appropriate stage first."
            )
        super().__init__(message)
