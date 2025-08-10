from functools import wraps


def View(url: str):

    def attr_decorator(_class):
        @wraps(_class)
        def wrapper(*args, **kwargs):
            return _class(*args, **kwargs)

        setattr(wrapper, 'url', url)

        return wrapper

    return attr_decorator