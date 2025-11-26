from django.urls import path
from . import views

app_name = 'viewer'

urlpatterns = [
    path('', views.index, name='index'),
    path('stream/', views.stream_view, name='stream'),
    path('code/', views.code_view, name='code'),
    path('memes/', views.memes_view, name='memes'),
    path('snapshots/', views.snapshots_view, name='snapshots'),
    path('download/requirements/', views.download_requirements, name='download_requirements'),
]
