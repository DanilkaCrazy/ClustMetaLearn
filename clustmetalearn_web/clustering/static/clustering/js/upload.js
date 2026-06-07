(function () {
    const KAGGLE_URL = /kaggle\.com\/datasets\/([^/]+)\/([^/?#]+)/i;
    const HF_URL = /huggingface\.co\/datasets\/([^/]+)\/([^/?#]+)/i;

    function parseSlug(input, urlRe) {
        const m = input.match(urlRe);
        if (m) return m[1] + '/' + m[2];
        if (/^[\w.-]+\/[\w.-]+$/.test(input.trim())) return input.trim();
        return null;
    }

    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('.upload-tab').forEach(function (tab) {
            tab.addEventListener('click', function () {
                const name = tab.getAttribute('data-tab');
                document.querySelectorAll('.upload-tab').forEach(function (t) {
                    t.classList.toggle('active', t.getAttribute('data-tab') === name);
                });
                document.querySelectorAll('.upload-panel').forEach(function (p) {
                    p.classList.toggle('active', p.id === 'panel-' + name);
                });
            });
        });

        const kaggleInput = document.getElementById('kaggle-ref-input');
        const kaggleDisplay = document.getElementById('kaggle-display-name');
        if (kaggleInput && kaggleDisplay) {
            kaggleInput.addEventListener('input', function () {
                const slug = parseSlug(kaggleInput.value, KAGGLE_URL);
                if (slug) {
                    kaggleDisplay.textContent = 'Dataset: ' + slug;
                    kaggleDisplay.classList.remove('hidden');
                } else {
                    kaggleDisplay.classList.add('hidden');
                }
            });
        }

        const hfInput = document.getElementById('hf-ref-input');
        const hfDisplay = document.getElementById('hf-display-name');
        if (hfInput && hfDisplay) {
            hfInput.addEventListener('input', function () {
                const slug = parseSlug(hfInput.value, HF_URL);
                if (slug) {
                    hfDisplay.textContent = 'Dataset: ' + slug;
                    hfDisplay.classList.remove('hidden');
                } else {
                    hfDisplay.classList.add('hidden');
                }
            });
        }

        ['kaggle-form', 'hf-form'].forEach(function (id) {
            const form = document.getElementById(id);
            if (!form) return;
            form.addEventListener('submit', function () {
                const progress = form.querySelector('[id$="-progress"]');
                if (progress) progress.classList.remove('hidden');
            });
        });
    });
})();
