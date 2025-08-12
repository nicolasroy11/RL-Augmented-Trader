from functools import wraps
import logging
db_logger = logging.getLogger('db')


def View(path: str, http_method: str, request_payload_type=None, return_type=None, query_params_type=None, description='No description', summary='No summary', group="General", include_in_swagger=False):
    """Sets view method attributes for use in various automations
        see https://stackoverflow.com/questions/338101/python-function-attributes-uses-and-abuses
        search: @with_attrs(counter=0, something='boing')
    """
    def attr_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                db_logger.exception(e)

        setattr(wrapper, 'path', path)
        setattr(wrapper, 'http_method', http_method)
        setattr(wrapper, 'name', fn.__name__)
        setattr(wrapper, 'view', fn)
        setattr(wrapper, 'path', path)
        setattr(wrapper, 'query_params_type', query_params_type)
        setattr(wrapper, 'description', description)
        setattr(wrapper, 'summary', summary)
        setattr(wrapper, 'group', group)
        setattr(wrapper, 'include_in_swagger', include_in_swagger)
        

        if request_payload_type != None: setattr(wrapper, 'request_payload_type', request_payload_type)
        if return_type != None: setattr(wrapper, 'return_type', return_type)

        return wrapper

    return attr_decorator
