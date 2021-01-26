from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about-model/', views.about_model, name='about_model'),
    path('about-data/', views.about_data, name='about_data')
]