import functools


def dict_to_kwargs(func):
    """Adapts a function to accept a single-dict argument of its arguments.

    Parameters:
        func (function): The function to be adapted.

    Returns:
        function: Same function that now accepts a single-dict argument.

    Raises:
        TypeError: If keyword arguments are used with the function.
        TypeError: If more than one argument is passed to the function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs:
            raise TypeError("Cannot use keyword arguments with this function.")
        if len(args) > 1:
            raise TypeError("This function only accepts a single dict argument.")
        return func(**args[0])

    return wrapper
