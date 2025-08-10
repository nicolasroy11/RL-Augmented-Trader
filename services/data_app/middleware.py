from typing import Any
from django.core.handlers.wsgi import WSGIRequest
from django.core.exceptions import BadRequest


class DataAppMiddleware(object):

    def __init__(self, get_response):
        """
        One-time configuration and initialisation.
        """
        self.get_response = get_response

    def __call__(self, request: WSGIRequest):
        """
        Code to be executed for each request before the view (and later
        middleware) are called.
        """
        try:
            response = self.get_response(request)
            if response.status_code == 404:
                pass
            return response
        except Exception as e:
            pass

    def process_view(self, request: WSGIRequest, view_func, view_args, view_kwargs):
        """
        Called just before Django calls the view.
        """

        path_segments = request.path.split('/')

        action = None
        if 'action' in view_kwargs:
            action = view_kwargs['action']
        elif 'action' in path_segments:
            action = path_segments[path_segments.index('action') + 1]

        return None

    def process_exception(self, request: WSGIRequest, exception: Any):
        """
        Called when a view raises an exception.
        """
        self.handle_exception(request=request, status_code=500, message=f"{exception.__class__.__name__}: {exception.args[0]}")
        return None

    def process_template_response(self, request, response):
        """
        Called just after the view has finished executing.
        """
        return response

    def handle_exception(self, request: WSGIRequest, status_code: int, message: str):
        e = Exception()
        e.args = f'{status_code}: {message}',
        raise e
