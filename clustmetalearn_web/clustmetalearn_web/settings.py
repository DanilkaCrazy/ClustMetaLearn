"""
Django settings for clustmetalearn_web project.
"""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-dev-key-change-in-production')
DEBUG = os.environ.get('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '*').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'clustering',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'clustering.middleware.UserPreferencesMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'clustmetalearn_web.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.i18n',
                'clustering.context_processors.design_colors',
                'clustering.context_processors.user_preferences',
                'clustering.context_processors.global_settings',
            ],
        },
    },
]

WSGI_APPLICATION = 'clustmetalearn_web.wsgi.application'

USE_POSTGRES = os.environ.get('USE_POSTGRES', 'False') == 'True'

if USE_POSTGRES:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.environ.get('POSTGRES_DB', 'clustmetalearn'),
            'USER': os.environ.get('POSTGRES_USER', 'postgres'),
            'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'postgres'),
            'HOST': os.environ.get('POSTGRES_HOST', 'db'),
            'PORT': os.environ.get('POSTGRES_PORT', '5432'),
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

LANGUAGES = [
    ('en', 'English'),
    ('ru', 'Russian'),
]

LOCALE_PATHS = [BASE_DIR / 'locale']

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

MODELS_DIR = BASE_DIR / 'models'

LOGIN_URL = 'clustering:login'
LOGIN_REDIRECT_URL = 'clustering:dashboard'
LOGOUT_REDIRECT_URL = 'clustering:index'

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER', 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = CELERY_BROKER_URL
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'

DARK_DESIGN_COLORS = {
    "background": "#121212",
    "surface": "#1e1e2f",
    "surface-dim": "#161622",
    "surface-bright": "#1e1e2f",
    "surface-container-lowest": "#1a1a28",
    "surface-container-low": "#222236",
    "surface-container": "#2a2a3e",
    "surface-container-high": "#323248",
    "surface-container-highest": "#3a3a52",
    "on-surface": "#e0e0e0",
    "on-surface-variant": "#b0b0c0",
    "inverse-surface": "#e0e0e0",
    "inverse-on-surface": "#121212",
    "outline": "#6b7280",
    "outline-variant": "#404058",
    "surface-tint": "#3b82f6",
    "primary": "#3b82f6",
    "on-primary": "#ffffff",
    "primary-container": "#2563eb",
    "on-primary-container": "#ffffff",
    "inverse-primary": "#93c5fd",
    "secondary": "#10b981",
    "on-secondary": "#ffffff",
    "secondary-container": "#059669",
    "on-secondary-container": "#ffffff",
    "tertiary": "#f87171",
    "on-tertiary": "#ffffff",
    "tertiary-container": "#ef4444",
    "on-tertiary-container": "#ffffff",
    "error": "#ef4444",
    "on-error": "#ffffff",
    "error-container": "#7f1d1d",
    "on-error-container": "#fecaca",
    "primary-fixed": "#1e3a5f",
    "primary-fixed-dim": "#2563eb",
    "on-primary-fixed": "#e0e0e0",
    "on-primary-fixed-variant": "#93c5fd",
    "secondary-fixed": "#064e3b",
    "secondary-fixed-dim": "#059669",
    "on-secondary-fixed": "#e0e0e0",
    "on-secondary-fixed-variant": "#6ee7b7",
    "tertiary-fixed": "#7f1d1d",
    "tertiary-fixed-dim": "#ef4444",
    "on-tertiary-fixed": "#e0e0e0",
    "on-tertiary-fixed-variant": "#fca5a5",
    "on-background": "#e0e0e0",
    "surface-variant": "#2a2a3e",
}

DESIGN_COLORS = {
    "surface": "#f8f9ff",
    "surface-dim": "#d0dbed",
    "surface-bright": "#f8f9ff",
    "surface-container-lowest": "#ffffff",
    "surface-container-low": "#eff4ff",
    "surface-container": "#e6eeff",
    "surface-container-high": "#dee9fc",
    "surface-container-highest": "#d9e3f6",
    "on-surface": "#121c2a",
    "on-surface-variant": "#424754",
    "inverse-surface": "#27313f",
    "inverse-on-surface": "#eaf1ff",
    "outline": "#727785",
    "outline-variant": "#c2c6d6",
    "surface-tint": "#005ac2",
    "primary": "#0058be",
    "on-primary": "#ffffff",
    "primary-container": "#2170e4",
    "on-primary-container": "#fefcff",
    "inverse-primary": "#adc6ff",
    "secondary": "#006c49",
    "on-secondary": "#ffffff",
    "secondary-container": "#6cf8bb",
    "on-secondary-container": "#00714d",
    "tertiary": "#b61722",
    "on-tertiary": "#ffffff",
    "tertiary-container": "#da3437",
    "on-tertiary-container": "#fffbff",
    "error": "#ba1a1a",
    "on-error": "#ffffff",
    "error-container": "#ffdad6",
    "on-error-container": "#93000a",
    "primary-fixed": "#d8e2ff",
    "primary-fixed-dim": "#adc6ff",
    "on-primary-fixed": "#001a42",
    "on-primary-fixed-variant": "#004395",
    "secondary-fixed": "#6ffbbe",
    "secondary-fixed-dim": "#4edea3",
    "on-secondary-fixed": "#002113",
    "on-secondary-fixed-variant": "#005236",
    "tertiary-fixed": "#ffdad7",
    "tertiary-fixed-dim": "#ffb3ad",
    "on-tertiary-fixed": "#410004",
    "on-tertiary-fixed-variant": "#930013",
    "background": "#f8f9ff",
    "on-background": "#121c2a",
    "surface-variant": "#d9e3f6",
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

USE_CELERY = os.environ.get('USE_CELERY', 'False') == 'True'
GITHUB_REPO_URL = 'https://github.com/DanilkaCrazy/ClustMetaLearn'
