from django.urls import path

from . import views

app_name = 'clustering'

urlpatterns = [
    path('', views.index, name='index'),
    path('demo/', views.demo_redirect, name='demo'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_view, name='upload'),
    path('recommend/<uuid:task_id>/', views.recommend_view, name='recommend'),
    path('api/task/<uuid:task_id>/meta/', views.task_meta_api, name='task_meta_api'),
    path('evolve/<uuid:task_id>/', views.evolve_view, name='evolve'),
    path('evolve/<uuid:task_id>/status/', views.evolve_status_view, name='evolve_status'),
    path('api/evolve/<uuid:task_id>/status/', views.evolve_status_api, name='evolve_status_api'),
    path('analyze/<uuid:task_id>/', views.analyze_view, name='analyze'),
    path('profile/', views.profile_view, name='profile'),
    path('about/', views.about_view, name='about'),
    path('docs/', views.docs_view, name='docs'),
    path('privacy/', views.privacy_view, name='privacy'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('set-theme/', views.set_theme, name='set_theme'),
    path('set-language/', views.set_language, name='set_language'),
    path('api/set-language/', views.set_language_api, name='set_language_api'),
    path('export/<uuid:task_id>/txt/', views.export_txt_view, name='export_txt'),
    path('export/<uuid:task_id>/pdf/', views.export_pdf_view, name='export_pdf'),
    path('export/all/txt/', views.export_txt_view, name='export_all_txt'),
    path('export/all/pdf/', views.export_pdf_view, name='export_all_pdf'),
]
