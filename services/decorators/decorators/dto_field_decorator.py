from functools import wraps
from typing import Any


def DtoField(default_value: Any = None, is_optional: bool = False):

    def attr_decorator(_class):
        @wraps(_class)
        def wrapper(*args, **kwargs):
            return _class(*args, **kwargs)

        setattr(wrapper, 'is_optional', is_optional)
        setattr(wrapper, 'default_value', default_value)

        return wrapper

    return attr_decorator