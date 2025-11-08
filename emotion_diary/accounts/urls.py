# accounts/urls.py
from django.urls import path
from . import views
from . import api_views
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
     path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),

    # JWT API endpoints
    path('api/register/', api_views.register_user, name='api_register'),
    path('api/login/', api_views.jwt_login, name='api_login'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
