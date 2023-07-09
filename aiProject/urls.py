
from os import name
from django.urls import path
from .views import (
    index,process_file,makePredict

)


urlpatterns = [
 
    path('',index,name="index"),
    path('process_file/', process_file, name='process_file'),
    path('make-prediction/', makePredict, name='makePredict'),
    
]