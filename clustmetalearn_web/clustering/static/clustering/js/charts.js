window.CMLCharts = {
    plotOptions: {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
        toImageButtonOptions: {
            format: 'png',
            filename: 'clustmetalearn_chart',
            height: 600,
            width: 900,
            scale: 2,
        },
    },

    renderAll(chartsData) {
        if (!window.Plotly || !chartsData) return;
        let charts;
        if (typeof chartsData === 'string') {
            try { charts = JSON.parse(chartsData); } catch (e) { console.error('Charts JSON parse error', e); return; }
        } else {
            charts = chartsData;
        }
        Object.keys(charts).forEach(function (key) {
            const el = document.getElementById('chart-' + key);
            if (!el || !charts[key]) return;
            const skeleton = el.querySelector('.skeleton');
            if (skeleton) skeleton.remove();
            try {
                let fig;
                if (typeof charts[key] === 'string') {
                    fig = JSON.parse(charts[key]);
                } else {
                    fig = charts[key];
                }
                const isDark = document.documentElement.classList.contains('dark');
                if (fig.layout) {
                    fig.layout.paper_bgcolor = 'rgba(0,0,0,0)';
                    fig.layout.plot_bgcolor = 'rgba(0,0,0,0)';
                    if (isDark) {
                        fig.layout.font = fig.layout.font || {};
                        fig.layout.font.color = '#e0e0e0';
                    }
                }
                Plotly.newPlot(el, fig.data, fig.layout, CMLCharts.plotOptions);
            } catch (e) { console.error('Chart error:', key, e); }
        });
    },

    initFromScript(scriptId) {
        const el = document.getElementById(scriptId);
        if (el) {
            CMLCharts.renderAll(el.textContent);
        }
    }
};

document.addEventListener('DOMContentLoaded', function () {
    CMLCharts.initFromScript('charts-data');
});
