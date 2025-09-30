from django.contrib import admin

# Register your models here.
from .models import Recipe


@admin.register(Recipe)
class RecipeAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "description", "time_minutes", "price")
    search_fields = ("title",)
    list_filter = ("time_minutes", "price")
    ordering = ("id",)
