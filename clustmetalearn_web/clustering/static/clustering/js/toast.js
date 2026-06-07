window.CMLToast = {
    show(message, type) {
        type = type || 'info';
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        const el = document.createElement('div');
        el.className = 'toast toast-' + type;
        el.textContent = message;
        container.appendChild(el);
        setTimeout(function () { el.remove(); }, 4000);
    }
};

document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('[data-copy]').forEach(function (btn) {
        btn.addEventListener('click', function () {
            const text = btn.getAttribute('data-copy');
            navigator.clipboard.writeText(text).then(function () {
                CMLToast.show(btn.getAttribute('data-copy-msg') || 'Copied!', 'success');
            });
        });
    });
});
