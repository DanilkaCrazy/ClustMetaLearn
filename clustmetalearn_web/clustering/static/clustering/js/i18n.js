(function () {
    const STORAGE_KEY = 'cml_language';

    function getCsrfToken() {
        const el = document.querySelector('[name=csrfmiddlewaretoken]');
        return el ? el.value : '';
    }

    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('#lang-form button[name="language"]').forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                if (!document.body.hasAttribute('data-auth')) return;
                e.preventDefault();
                const lang = btn.value;
                localStorage.setItem(STORAGE_KEY, lang);
                fetch('/api/set-language/?language=' + encodeURIComponent(lang), {
                    credentials: 'same-origin',
                    headers: { 'X-Requested-With': 'XMLHttpRequest' },
                }).then(function (r) { return r.json(); }).then(function (data) {
                    if (data.reload) window.location.reload();
                }).catch(function () {
                    btn.closest('form').submit();
                });
            });
        });

        const langSelect = document.querySelector('#profile-lang-form select[name="language"]');
        if (langSelect) {
            langSelect.addEventListener('change', function () {
                localStorage.setItem(STORAGE_KEY, langSelect.value);
            });
        }
    });
})();
