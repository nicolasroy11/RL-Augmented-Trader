from service.service_app.views import current_datetime

from django.urls import path
from service.service_app.views import current_datetime

urlpatterns = [
    path('current_datetime', current_datetime, name='current_datetime'),
]