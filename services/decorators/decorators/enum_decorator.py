from functools import wraps


def cgEnum():

    def attr_decorator(_class):
        @wraps(_class)
        def wrapper(*args, **kwargs):
            return _class(*args, **kwargs)

        return _class

    return attr_decorator