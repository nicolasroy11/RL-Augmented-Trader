from functools import wraps

class cg:
    def Dto():

        def attr_decorator(_class):
            @wraps(_class)
            def wrapper(*args, **kwargs):
                return _class(*args, **kwargs)

            # setattr(wrapper, 'class_name', class_name)

            return wrapper

        return attr_decorator

    def Http(path: str, http_method: str, request_payload_type = None, return_type = None):
        """Sets view method attributes for use in various automations
            see https://stackoverflow.com/questions/338101/python-function-attributes-uses-and-abuses
            search: @with_attrs(counter=0, something='boing')
        """
        def attr_decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            setattr(wrapper, 'path', path)
            setattr(wrapper, 'http_method', http_method)
            setattr(wrapper, 'name', fn.__name__)
            setattr(wrapper, 'view', fn)
            setattr(wrapper, 'path', path)

            if request_payload_type != None: setattr(wrapper, 'request_payload_type', request_payload_type)
            if return_type != None: setattr(wrapper, 'return_type', return_type)

            return wrapper

        return attr_decorator

    def View(url: str):

        def attr_decorator(_class):
            @wraps(_class)
            def wrapper(*args, **kwargs):
                return _class(*args, **kwargs)

            setattr(wrapper, 'url', url)

            return wrapper

        return attr_decorator