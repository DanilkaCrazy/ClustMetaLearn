from django.contrib import admin

from .models import (
    ClusteringTask, DatasetAnalysis, EvolutionarySession,
    GenerationLog, UserProfile,
)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'theme_preference', 'language', 'updated_at')


@admin.register(ClusteringTask)
class ClusteringTaskAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'user', 'status', 'recommended_algorithm', 'created_at')
    list_filter = ('status', 'input_format')
    search_fields = ('file_name', 'user__username')


@admin.register(DatasetAnalysis)
class DatasetAnalysisAdmin(admin.ModelAdmin):
    list_display = ('dataset_name', 'user', 'recommended_algorithm', 'created_at')


@admin.register(EvolutionarySession)
class EvolutionarySessionAdmin(admin.ModelAdmin):
    list_display = ('task', 'status', 'current_generation', 'total_generations')


@admin.register(GenerationLog)
class GenerationLogAdmin(admin.ModelAdmin):
    list_display = ('session', 'generation_number', 'fitness_score', 'timestamp')
