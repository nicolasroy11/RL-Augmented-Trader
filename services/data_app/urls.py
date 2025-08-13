from rest_framework.schemas import get_schema_view
import inspect
from django.urls import path, re_path
from rest_framework.schemas import get_schema_view
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from drf_yasg.generators import OpenAPISchemaGenerator
from drf_yasg.utils import swagger_auto_schema
from rest_framework.decorators import api_view
from services.data_app.views.data_view import Ticks


class BothHttpAndHttpsSchemaGenerator(OpenAPISchemaGenerator):
    def get_schema(self, request=None, public=True):
        schema = super().get_schema(request, public)
        schema.schemes = ["https", "http"]
        return schema


schema_view = get_schema_view(
    openapi.Info(
        title="Ticks API",
        default_version='v1',
        description="<b>Endpoints for the tick data API</b>",
    ),
    public=True,
    generator_class=BothHttpAndHttpsSchemaGenerator,
    permission_classes=[permissions.AllowAny],
)
urlpatterns = [
    re_path(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    re_path(r'^swagger/$', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    re_path(r'^redoc/$', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]

# generate urlpatterns from each method in the view class
grids_method_list = inspect.getmembers(Ticks, predicate=inspect.isfunction)
for i, method in grids_method_list:

    view = method.view
    if method.include_in_swagger:
        api_decorated_view = api_view([method.http_method])(method.view)
        swagger_view = swagger_auto_schema(
            method=method.http_method,
            responses={200: method.return_type},
            query_serializer=method.query_params_type,
            operation_id=method.summary,
            operation_description=method.description,
            tags=[method.group]
        )(api_decorated_view)
        view = swagger_view

    # we ignore the methods that do not have the http_method, which is
    # only present for those decorated with the custom @View decorator
    urlpatterns.append(path(Ticks.url + method.path, view))