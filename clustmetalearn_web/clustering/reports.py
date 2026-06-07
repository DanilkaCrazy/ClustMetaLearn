"""Generate TXT and PDF experiment reports."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Iterable

from django.utils.translation import gettext as _

from .models import ClusteringTask, EvolutionarySession
from .utils import format_algorithm


MAX_EXPORT_TASKS = 100


def _format_meta_table(meta: dict) -> str:
    lines = ['Feature\tValue']
    for key, val in sorted(meta.items()):
        try:
            lines.append(f'{key}\t{float(val):.6f}')
        except (TypeError, ValueError):
            lines.append(f'{key}\t{val}')
    return '\n'.join(lines)


def _evolution_section(session: EvolutionarySession | None) -> list[str]:
    if not session:
        return [_('Evolution: not run')]
    lines = [
        _('Evolution status: %(status)s') % {'status': session.get_status_display()},
        _('Best fitness: %(v)s') % {'v': f'{session.best_fitness:.4f}'},
        _('Best pipeline: %(p)s') % {'p': session.best_pipeline or 'N/A'},
    ]
    gens = session.generations.order_by('-generation_number')[:5]
    if gens:
        lines.append(_('Last 5 generations:'))
        for g in reversed(list(gens)):
            lines.append(
                f"  gen {g.generation_number}: fitness={g.fitness_score:.4f}, "
                f"pipeline={g.best_pipeline}"
            )
    return lines


def build_single_report_text(task: ClusteringTask) -> str:
    """Plain-text report for one experiment."""
    sep = '=' * 60
    meta = task.meta_features_json or {}
    hp = task.hp_intervals or {}
    session = EvolutionarySession.objects.filter(task=task).first()

    lines = [
        sep,
        _('ClustMetaLearn Experiment Report'),
        sep,
        _('Dataset: %(name)s') % {'name': task.file_name},
        _('Upload date: %(d)s') % {'d': task.created_at.strftime('%Y-%m-%d %H:%M')},
        _('Format: %(f)s') % {'f': task.input_format},
        _('Status: %(s)s') % {'s': task.get_status_display()},
        '',
        _('--- Meta-features ---'),
        _format_meta_table(meta),
        '',
        _('--- Recommendations ---'),
        _('Metric: %(m)s') % {'m': task.recommended_metric or 'N/A'},
        _('Algorithm: %(a)s') % {
            'a': format_algorithm(task.recommended_algorithm) if task.recommended_algorithm else 'N/A',
        },
        _('Top-3: %(t)s') % {'t': ', '.join(task.top3_algorithms or [])},
        _('Expected ARI: %(a)s') % {
            'a': f'{task.ari_prediction:.4f}' if task.ari_prediction is not None else 'N/A',
        },
        '',
        _('--- Hyperparameters ---'),
        f"k_min={hp.get('k_min', 'N/A')}",
        f"k_max={hp.get('k_max', 'N/A')}",
        f"k_median={hp.get('k_median', 'N/A')}",
        f"scaler_rate={hp.get('scaler_rate', 'N/A')}",
        '',
        _('--- Evolution ---'),
    ]
    lines.extend(_evolution_section(session))
    lines.append(sep)
    return '\n'.join(lines)


def build_all_report_text(tasks: Iterable[ClusteringTask]) -> str:
    """Summary report for multiple experiments."""
    task_list = list(tasks)
    if len(task_list) > MAX_EXPORT_TASKS:
        raise ValueError(
            _('Too many experiments (%(n)s). Maximum is %(max)s.') % {
                'n': len(task_list), 'max': MAX_EXPORT_TASKS,
            }
        )

    sep = '=' * 60
    success = [t for t in task_list if t.status == 'success']
    failed = [t for t in task_list if t.status == 'failed']
    avg_ari = None
    aris = [t.ari_prediction for t in success if t.ari_prediction is not None]
    if aris:
        avg_ari = sum(aris) / len(aris)

    lines = [
        sep,
        _('ClustMetaLearn — All Experiments Report'),
        sep,
        _('Generated: %(d)s') % {'d': datetime.now().strftime('%Y-%m-%d %H:%M')},
        _('Total experiments: %(n)s') % {'n': len(task_list)},
        _('Successful: %(n)s') % {'n': len(success)},
        _('Failed: %(n)s') % {'n': len(failed)},
        _('Average ARI: %(a)s') % {
            'a': f'{avg_ari:.4f}' if avg_ari is not None else 'N/A',
        },
        '',
        _('--- Summary Table ---'),
        'ID\tDate\tDataset\tAlgorithm\tMetric\tARI\tStatus',
    ]
    for t in task_list:
        lines.append(
            f"{t.id}\t{t.created_at.strftime('%Y-%m-%d')}\t{t.file_name}\t"
            f"{t.recommended_algorithm or 'N/A'}\t{t.recommended_metric or 'N/A'}\t"
            f"{t.ari_prediction if t.ari_prediction is not None else 'N/A'}\t"
            f"{t.status}"
        )

    if aris:
        lines.extend([
            '',
            _('--- ARI Trend ---'),
            _('ARI values over time (oldest to newest):'),
            ', '.join(f'{a:.3f}' for a in aris),
        ])
    lines.append(sep)
    return '\n'.join(lines)


def build_pdf_from_html(html_content: str) -> bytes:
    """Convert HTML report to PDF bytes using weasyprint."""
    try:
        from weasyprint import HTML
    except ImportError as exc:
        raise ImportError(
            'weasyprint is not installed. Add it to requirements and pip install.'
        ) from exc
    return HTML(string=html_content).write_pdf()


def render_report_html(title: str, body_sections: list[str], chart_png: bytes | None = None) -> str:
    """Minimal HTML layout for PDF export."""
    import base64

    chart_block = ''
    if chart_png:
        b64 = base64.b64encode(chart_png).decode('ascii')
        chart_block = f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:1em 0;" />'

    sections_html = ''.join(
        f'<section style="margin-bottom:1.5em;"><pre style="white-space:pre-wrap;font-family:monospace;font-size:11px;">{s}</pre></section>'
        for s in body_sections
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin: 2em; color: #121c2a; }}
h1 {{ color: #0058be; border-bottom: 2px solid #0058be; padding-bottom: 0.3em; }}
</style></head>
<body><h1>{title}</h1>{chart_block}{sections_html}</body></html>"""


def ari_trend_png(tasks: list[ClusteringTask]) -> bytes | None:
    """Generate small ARI trend chart as PNG for all-experiments PDF."""
    aris = [
        (t.created_at, t.ari_prediction)
        for t in tasks
        if t.ari_prediction is not None
    ]
    if len(aris) < 2:
        return None
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(
            x=[a[0] for a in aris],
            y=[a[1] for a in aris],
            mode='lines+markers',
        ))
        fig.update_layout(title='ARI Trend', height=300, width=600, template='plotly_white')
        return fig.to_image(format='png', engine='kaleido')
    except Exception:
        return None
