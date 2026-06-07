import json

from django.conf import settings


def design_colors(request):
    theme = request.session.get('theme', 'light')
    if request.user.is_authenticated:
        try:
            theme = request.user.profile.theme_preference
        except Exception:
            pass
    colors = settings.DARK_DESIGN_COLORS if theme == 'dark' else settings.DESIGN_COLORS
    return {
        'DESIGN_COLORS_JSON': json.dumps(colors),
        'DESIGN_COLORS': colors,
        'is_dark_theme': theme == 'dark',
    }


def user_preferences(request):
    theme = request.session.get('theme', 'light')
    language = request.session.get(
        settings.LANGUAGE_COOKIE_NAME, settings.LANGUAGE_CODE,
    )
    if request.user.is_authenticated:
        try:
            profile = request.user.profile
            theme = profile.theme_preference
            language = profile.language
        except Exception:
            pass
    return {
        'user_theme': theme,
        'user_language': language,
    }


def global_settings(request):
    return {
        'GITHUB_REPO_URL': settings.GITHUB_REPO_URL,
    }
