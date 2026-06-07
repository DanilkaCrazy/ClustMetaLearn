from django.conf import settings
from django.utils import translation


class UserPreferencesMiddleware:
    """Activate language from user profile or session after auth is available."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        lang = request.session.get(settings.LANGUAGE_COOKIE_NAME)
        if not lang:
            lang = request.COOKIES.get(settings.LANGUAGE_COOKIE_NAME)
        if request.user.is_authenticated:
            try:
                lang = request.user.profile.language
            except Exception:
                pass
        if lang and lang in dict(settings.LANGUAGES):
            translation.activate(lang)
            request.LANGUAGE_CODE = lang
        response = self.get_response(request)
        if lang and lang in dict(settings.LANGUAGES):
            response.set_cookie(settings.LANGUAGE_COOKIE_NAME, lang)
        return response
