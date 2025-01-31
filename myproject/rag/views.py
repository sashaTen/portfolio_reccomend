from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

from zenml.client import Client



def hello(request):
   # simple_ml_pipeline()

   artifact = Client().get_artifact_version("44a244a1-9955-43b3-bc19-45dd3a4a2342")
   data = artifact.load()
   return HttpResponse(data)


