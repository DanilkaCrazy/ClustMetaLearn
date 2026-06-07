import uuid

from django.contrib.auth.models import User
from django.db import models
from django.utils.translation import gettext_lazy as _


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    phone_number = models.CharField(max_length=20, null=True, blank=True)
    theme_preference = models.CharField(
        max_length=10, default='light',
        choices=[('light', 'Light'), ('dark', 'Dark')],
    )
    company_org = models.CharField(max_length=255, null=True, blank=True)
    language = models.CharField(
        max_length=5, default='en',
        choices=[('en', 'English'), ('ru', 'Russian')],
    )
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s profile"


class ClusteringTask(models.Model):
    STATUS_CHOICES = [
        ('pending', _('Pending')),
        ('processing', _('Processing')),
        ('success', _('Success')),
        ('failed', _('Failed')),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='tasks',
        null=True, blank=True,
    )
    SOURCE_CHOICES = [
        ('file', _('File upload')),
        ('kaggle', _('Kaggle')),
        ('huggingface', _('Hugging Face')),
    ]
    file_name = models.CharField(max_length=255)
    data_source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='file')
    source_identifier = models.CharField(max_length=500, blank=True)
    input_format = models.CharField(max_length=10, default='csv')
    n_samples = models.IntegerField(null=True, blank=True)
    n_features = models.IntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True)

    meta_features_json = models.JSONField(default=dict)
    recommended_metric = models.CharField(max_length=100, blank=True)
    recommended_algorithm = models.CharField(max_length=100, blank=True)
    top3_algorithms = models.JSONField(default=list)
    hp_intervals = models.JSONField(default=dict)
    predicted_cvi_metric = models.CharField(max_length=100, blank=True)
    ari_prediction = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def get_meta_features(self):
        return self.meta_features_json

    def __str__(self):
        return f"{self.file_name} ({self.status})"


class DatasetAnalysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analyses')
    dataset_file = models.FileField(upload_to='datasets/')
    dataset_name = models.CharField(max_length=255)
    file_format = models.CharField(max_length=10, default='csv')
    n_samples = models.IntegerField(null=True, blank=True)
    n_features = models.IntegerField(null=True, blank=True)
    meta_features = models.JSONField(default=dict)
    recommended_metric = models.CharField(max_length=100, null=True, blank=True)
    recommended_algorithm = models.CharField(max_length=100, null=True, blank=True)
    top3_algorithms = models.JSONField(default=list)
    hp_intervals = models.JSONField(default=dict)
    predicted_cvi_metric = models.CharField(max_length=100, null=True, blank=True)
    ari_prediction = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class EvolutionarySession(models.Model):
    STATUS_CHOICES = [
        ('pending', _('Pending')),
        ('running', _('Running')),
        ('completed', _('Completed')),
        ('failed', _('Failed')),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.OneToOneField(
        ClusteringTask, on_delete=models.CASCADE, related_name='evolution',
        null=True, blank=True,
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    current_generation = models.IntegerField(default=0)
    total_generations = models.IntegerField(default=10)
    population_size = models.IntegerField(default=40)
    fitness_metric = models.CharField(max_length=50, default='silhouette')
    bandit_strategy = models.CharField(max_length=20, default='ucb1')
    best_fitness = models.FloatField(default=0.0)
    best_pipeline = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class GenerationLog(models.Model):
    session = models.ForeignKey(
        EvolutionarySession, on_delete=models.CASCADE, related_name='generations',
    )
    generation_number = models.IntegerField()
    best_pipeline = models.CharField(max_length=255)
    fitness_score = models.FloatField()
    pipelines_evaluated = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)
