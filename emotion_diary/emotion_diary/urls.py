from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render
from memories import views
def index_redirect(request):
    # optional: redirect root to add memory page
    return render(request, 'index.html')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('memories/', include('memories.urls')),
    path('memories/add/', views.add_memory, name='add_memory'),
    path('', views.add_memory, name='home'),
    path('', index_redirect, name='root_index'),
    path('accounts/', include('accounts.urls')),
]