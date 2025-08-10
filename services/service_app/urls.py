from services.service_app.views import current_datetime

from django.urls import path
from services.service_app.views import current_datetime

urlpatterns = [
    path('current_datetime', current_datetime, name='current_datetime'),
]