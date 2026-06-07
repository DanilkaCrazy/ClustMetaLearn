"""Plotly chart builders for analysis and dashboard pages."""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def _template(dark: bool = False) -> str:
    return 'plotly_dark' if dark else 'plotly_white'


def _fig_json(fig) -> str:
    return pio.to_json(fig)


def _layout_defaults(dark: bool, height: int = 380) -> dict:
    layout = {
        'template': _template(dark),
        'height': height,
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#e0e0e0' if dark else '#121c2a'},
        'margin': dict(l=50, r=20, t=50, b=80),
    }
    return layout


def meta_histogram_chart(meta: dict, dark: bool = False) -> str:
    keys = ['n_samples', 'n_features', 'skewness', 'kurtosis', 'mean', 'std', 'var', 'entropy']
    keys = [k for k in keys if k in meta] or list(meta.keys())[:12]
    values = [float(meta[k]) for k in keys]
    fig = go.Figure(go.Bar(
        x=keys, y=values,
        marker_color='#3b82f6' if dark else '#0058be',
        hovertemplate='%{x}: %{y:.4f}<extra></extra>',
    ))
    layout = _layout_defaults(dark, 380)
    layout['title'] = 'Meta-features Distribution'
    layout['xaxis_tickangle'] = -45
    fig.update_layout(**layout)
    return _fig_json(fig)


def meta_correlation_heatmap(meta: dict, all_metas: list[dict] | None = None, dark: bool = False) -> str:
    keys = list(meta.keys())
    if all_metas and len(all_metas) >= 2:
        matrix = np.array([[float(m.get(k, 0)) for k in keys] for m in all_metas])
        data = np.corrcoef(matrix)
        title = 'Meta-features Correlation (across tasks)'
    elif len(keys) >= 2:
        vals = np.array([float(meta[k]) for k in keys])
        normed = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        data = np.outer(normed, normed)
        title = 'Meta-features Co-occurrence (normalized)'
    else:
        fig = go.Figure()
        fig.add_annotation(text='Not enough features', showarrow=False)
        fig.update_layout(**_layout_defaults(dark, 300))
        return _fig_json(fig)
    fig = go.Figure(go.Heatmap(
        z=data, x=keys, y=keys if data.ndim == 2 else keys,
        colorscale='Blues', zmin=-1 if all_metas else 0, zmax=1,
    ))
    layout = _layout_defaults(dark, 420)
    layout['title'] = title
    layout['margin'] = dict(l=80, r=20, t=50, b=120)
    fig.update_layout(**layout)
    return _fig_json(fig)


def pca_scatter_chart(meta: dict, algorithm: str = '', dark: bool = False) -> str:
    pc1 = float(meta.get('pca_var_1', 0))
    pc2 = float(meta.get('pca_var_2', 0))
    fig = go.Figure(go.Scatter(
        x=[pc1], y=[pc2],
        mode='markers+text',
        marker=dict(size=18, color='#10b981' if dark else '#006c49'),
        text=[algorithm or 'current'],
        textposition='top center',
        name='Current dataset',
    ))
    layout = _layout_defaults(dark, 360)
    layout['title'] = 'PCA Variance Space (PC1 vs PC2)'
    layout['xaxis_title'] = 'PC1 variance'
    layout['yaxis_title'] = 'PC2 variance'
    fig.update_layout(**layout)
    return _fig_json(fig)


def top3_bar_chart(probs: dict, dark: bool = False) -> str:
    if not probs:
        fig = go.Figure()
        fig.add_annotation(text='No probability data', showarrow=False)
        fig.update_layout(**_layout_defaults(dark, 300))
        return _fig_json(fig)
    labels = list(probs.keys())
    values = list(probs.values())
    colors = ['#3b82f6', '#10b981', '#f87171'] if dark else ['#0058be', '#006c49', '#b61722']
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors[:len(labels)],
        text=[f'{v:.1%}' for v in values],
        textposition='auto',
    ))
    layout = _layout_defaults(dark, 340)
    layout['title'] = 'Top-3 Algorithm Probabilities'
    layout['yaxis_tickformat'] = '.0%'
    fig.update_layout(**layout)
    return _fig_json(fig)


def ari_scatter_chart(tasks_data: list[dict], dark: bool = False) -> str:
    if len(tasks_data) < 1:
        fig = go.Figure()
        fig.add_annotation(text='No ARI data yet', showarrow=False)
        fig.update_layout(**_layout_defaults(dark, 300))
        return _fig_json(fig)
    fig = go.Figure(go.Scatter(
        x=list(range(len(tasks_data))),
        y=[t['ari'] for t in tasks_data],
        mode='markers+lines' if len(tasks_data) >= 2 else 'markers',
        marker=dict(size=10, color='#3b82f6' if dark else '#0058be'),
        text=[t['name'] for t in tasks_data],
        hovertemplate='%{text}<br>ARI: %{y:.3f}<extra></extra>',
    ))
    layout = _layout_defaults(dark, 340)
    layout['title'] = 'Predicted ARI Across Tasks'
    layout['xaxis_title'] = 'Task index'
    layout['yaxis_title'] = 'Predicted ARI'
    fig.update_layout(**layout)
    return _fig_json(fig)


def evolution_placeholder_chart(generations: list[dict] | None = None, dark: bool = False) -> str:
    if generations:
        xs = [g['generation'] for g in generations]
        ys = [g['fitness'] for g in generations]
        title = 'Evolution Quality'
    else:
        xs = list(range(1, 11))
        ys = [0.3 + 0.05 * i + np.random.default_rng(42).normal(0, 0.01) for i in xs]
        title = 'Evolution Quality (no data yet)'
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode='lines+markers',
        line=dict(color='#3b82f6' if dark else '#0058be', width=2),
        marker=dict(size=8),
    ))
    layout = _layout_defaults(dark, 340)
    layout['title'] = title
    layout['xaxis_title'] = 'Generation'
    layout['yaxis_title'] = 'Best fitness'
    fig.update_layout(**layout)
    return _fig_json(fig)


def dashboard_daily_chart(daily_counts: dict, dark: bool = False) -> str:
    dates = list(daily_counts.keys())
    counts = list(daily_counts.values())
    fig = go.Figure(go.Bar(
        x=dates, y=counts,
        marker_color='#3b82f6' if dark else '#2170e4',
    ))
    layout = _layout_defaults(dark, 260)
    layout['title'] = 'Tasks per Day'
    layout['margin'] = dict(l=40, r=20, t=40, b=40)
    fig.update_layout(**layout)
    return _fig_json(fig)
