from rest_framework import serializers
from .models import Recipe, RecipeRating

class RecipeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recipe
        fields = ["id", "title", "description", "time_minutes", "price"]

class RecipeRatingSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecipeRating
        fields = ["id", "stars", "created_at"]
        read_only_fields = ["id", "created_at"]

    def validate_stars(self, value):
        if not (1 <= value <= 5):
            raise serializers.ValidationError("Rating must be between 1 and 5.")
        return value