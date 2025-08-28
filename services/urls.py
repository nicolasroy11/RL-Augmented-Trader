"""
URL configuration for traderplus project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from data_app import urls as data_app_urls
from trainer_app import urls as rl_app_urls
from trader_app import urls as trader_app_urls
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include(data_app_urls.urlpatterns  + rl_app_urls.urlpatterns + trader_app_urls.urlpatterns ))
]
