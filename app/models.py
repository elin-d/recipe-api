from django.db import models


class Recipe(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    time_minutes = models.IntegerField()
    price = models.DecimalField(max_digits=5, decimal_places=2)

    def __str__(self):
        return self.title
