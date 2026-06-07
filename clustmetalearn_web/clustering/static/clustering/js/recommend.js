document.addEventListener('DOMContentLoaded', function () {
    const taskId = document.getElementById('recommend-root');
    if (!taskId) return;
    const id = taskId.dataset.taskId;
    const refreshBtn = document.getElementById('refresh-meta');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function () {
            refreshBtn.disabled = true;
            fetch('/api/task/' + id + '/meta/')
                .then(r => r.json())
                .then(function (data) {
                    const tbody = document.getElementById('meta-table-body');
                    if (tbody && data.meta_features) {
                        tbody.innerHTML = '';
                        Object.entries(data.meta_features).forEach(function ([k, v]) {
                            tbody.innerHTML += '<tr class="hover:bg-surface-container-low/50"><td class="p-2 font-mono text-sm">' + k + '</td><td class="p-2 font-mono text-sm">' + Number(v).toFixed(4) + '</td></tr>';
                        });
                    }
                    CMLToast.show(refreshBtn.dataset.successMsg || 'Updated', 'success');
                })
                .catch(function () { CMLToast.show('Error', 'error'); })
                .finally(function () { refreshBtn.disabled = false; });
        });
    }
});
