from django.urls import path
from . import views
from .views import add_memory, view_memories, search_summary, api_add_memory, api_search_memories

urlpatterns = [
     path('', add_memory, name='add_memory'),
    path('view/', view_memories, name='view_memories'),
    path('search/', search_summary, name='search_summary'),

    # 🔐 API Endpoints
    path('api/add/', api_add_memory, name='api_add_memory'),
    path('api/search/', api_search_memories, name='api_search_memories'),
]
