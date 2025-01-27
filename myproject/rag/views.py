from django.shortcuts import render
from .rag_example import simple_ml_pipeline
# Create your views here.
from django.http import HttpResponse




def hello(request):
   # simple_ml_pipeline()
    return HttpResponse('dockers ')


