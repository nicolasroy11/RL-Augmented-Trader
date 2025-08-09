from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
import datetime
from service.service_app.apps import current_trade_cycle


def current_datetime(request):
    now = datetime.datetime.now()
    html = '<html lang="en"><body>It is now %s.</body></html>' % now
    ctc = current_trade_cycle
    return HttpResponse(html)