from django.urls import path
from .views import get_active_announcements

urlpatterns = [
    path('list/', get_active_announcements, name='get_active_announcements'),
]