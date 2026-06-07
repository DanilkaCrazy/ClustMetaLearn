document.addEventListener('DOMContentLoaded', function () {
    const root = document.getElementById('evolve-status-root');
    if (!root) return;
    const apiUrl = root.dataset.apiUrl;
    const bar = document.getElementById('progress-fill');
    const statusEl = document.getElementById('evolve-status-text');
    const genEl = document.getElementById('evolve-gen-text');
    const pipelineEl = document.getElementById('evolve-pipeline');

    function poll() {
        fetch(apiUrl).then(r => r.json()).then(function (data) {
            if (bar) bar.style.width = data.progress + '%';
            if (statusEl) statusEl.textContent = data.status;
            if (genEl) genEl.textContent = data.current_generation + ' / ' + data.total_generations;
            if (pipelineEl && data.best_pipeline) pipelineEl.textContent = data.best_pipeline;
            if (data.status === 'running' || data.status === 'pending') {
                setTimeout(poll, 1500);
            }
        });
    }
    poll();
});
