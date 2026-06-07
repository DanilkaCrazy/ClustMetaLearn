(function () {
    const STORAGE_KEY = 'cml_theme';

    function applyTheme(theme) {
        document.documentElement.classList.remove('light', 'dark');
        document.documentElement.classList.add(theme);
        document.documentElement.setAttribute('data-theme', theme);
    }

    function getStoredTheme() {
        return localStorage.getItem(STORAGE_KEY);
    }

    function setStoredTheme(theme) {
        localStorage.setItem(STORAGE_KEY, theme);
    }

    document.addEventListener('DOMContentLoaded', function () {
        const serverTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const isAuth = document.body.hasAttribute('data-auth');

        if (isAuth) {
            applyTheme(serverTheme);
            setStoredTheme(serverTheme);
        } else {
            const stored = getStoredTheme();
            applyTheme(stored && stored !== serverTheme ? stored : serverTheme);
            if (!stored) setStoredTheme(serverTheme);
        }

        document.querySelectorAll('#theme-form button[name="theme"]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                setStoredTheme(btn.value);
                applyTheme(btn.value);
            });
        });
    });
})();
