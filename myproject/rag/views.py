from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

from zenml.client import Client



def hello(request):
   # simple_ml_pipeline()

   return HttpResponse('helloi ')


