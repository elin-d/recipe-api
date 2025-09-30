from rest_framework import viewsets

from .models import Recipe
from .serializers import RecipeSerializer


class RecipeViewSet(viewsets.ModelViewSet):
    queryset = Recipe.objects.all().order_by("id")
    serializer_class = RecipeSerializer

