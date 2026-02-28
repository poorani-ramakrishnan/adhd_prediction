from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict, name='predict'),
    path("game/", views.game, name="game"),
    path("download-report/", views.download_report, name="download_report"),
]
