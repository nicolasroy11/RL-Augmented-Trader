from functools import wraps
from enum import Enum

class TsTypes(Enum):
    Interface = 'interface'
    Class = 'class'


def Dto( ts_type: TsTypes = TsTypes.Interface ):

    def attr_decorator(_class):
        @wraps(_class)
        def wrapper(*args, **kwargs):
            return _class(*args, **kwargs)

        setattr(wrapper, 'ts_type', ts_type)

        return wrapper

    return attr_decorator