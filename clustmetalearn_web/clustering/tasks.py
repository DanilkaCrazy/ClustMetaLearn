import os
import time

from celery import shared_task
from django.conf import settings

from .models import ClusteringTask, EvolutionarySession, GenerationLog
from .utils import extract_meta_features, predict_clustering_strategy


def _try_run_real_evolution(session, file_path):
    """
    Attempt to call clustmetalearn.tpot_clustering.evolve.run_evolution.
    Returns True if real evolution ran, False to fall back to stub.
    """
    try:
        import numpy as np
        import pandas as pd
        from clustmetalearn.tpot_clustering.evolve import run_evolution
    except ImportError:
        return False

    task = session.task
    if task.input_format == 'bin':
        data = np.fromfile(
            file_path, dtype=np.float32,
        ).reshape((task.n_samples, task.n_features))
        X = data
    else:
        df = pd.read_csv(file_path)
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return False
        X = df[num_cols].to_numpy()

    config = {
        'generations': session.total_generations,
        'population_size': session.population_size,
        'fitness_metric': session.fitness_metric,
        'bandit_strategy': session.bandit_strategy,
    }
    result = run_evolution(X, y_eval=None, space=None, config=config)
    best_fitness = float(result.get('best_fitness', 0))
    best_pipeline = str(result.get('best_pipeline', ''))
    history = result.get('history', [])

    session.best_fitness = best_fitness
    session.best_pipeline = best_pipeline
    for i, entry in enumerate(history, start=1):
        session.current_generation = i
        session.save(update_fields=['current_generation', 'best_fitness', 'best_pipeline'])
        GenerationLog.objects.create(
            session=session,
            generation_number=i,
            best_pipeline=str(entry.get('pipeline', best_pipeline)),
            fitness_score=float(entry.get('fitness', best_fitness)),
            pipelines_evaluated=session.population_size,
        )
    return True


def _run_evolution_stub(session):
    """Emulate evolutionary search with incremental progress logs."""
    for gen in range(1, session.total_generations + 1):
        time.sleep(0.5)
        fitness = 0.3 + 0.04 * gen
        session.current_generation = gen
        session.best_fitness = fitness
        session.best_pipeline = f'scaler|{session.task.recommended_algorithm}|k={gen + 2}'
        session.save(update_fields=['current_generation', 'best_fitness', 'best_pipeline'])
        GenerationLog.objects.create(
            session=session,
            generation_number=gen,
            best_pipeline=session.best_pipeline,
            fitness_score=fitness,
            pipelines_evaluated=session.population_size,
        )


@shared_task(bind=True)
def run_evolution_task(self, session_id):
    """Run evolutionary search (real library or stub with progress emulation)."""
    session = EvolutionarySession.objects.select_related('task').get(id=session_id)
    session.status = 'running'
    session.current_generation = 0
    session.save(update_fields=['status', 'current_generation'])

    file_path = None
    try:
        datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
        prefix = f'task_{session.task.id}_'
        for name in os.listdir(datasets_dir):
            if name.startswith(prefix):
                file_path = os.path.join(datasets_dir, name)
                break

        used_real = False
        if file_path and os.path.exists(file_path):
            used_real = _try_run_real_evolution(session, file_path)

        if not used_real:
            _run_evolution_stub(session)

        session.status = 'completed'
        session.save(update_fields=['status'])
    except Exception:
        session.status = 'failed'
        session.save(update_fields=['status'])
        raise


@shared_task
def run_async_pipeline(task_id, file_path):
    task = ClusteringTask.objects.get(id=task_id)
    try:
        task.status = 'processing'
        task.save(update_fields=['status'])

        meta = extract_meta_features(
            file_path,
            fmt=task.input_format,
            n_samples=task.n_samples,
            n_features=task.n_features,
        )
        metric, algo, ari, top3, hp, top3_probs = predict_clustering_strategy(meta)

        task.meta_features_json = meta
        task.recommended_metric = metric
        task.recommended_algorithm = algo
        task.predicted_cvi_metric = metric
        task.ari_prediction = ari
        task.top3_algorithms = top3
        task.hp_intervals = hp
        task.status = 'success'
        task.save()
    except Exception as e:
        task.status = 'failed'
        task.error_message = str(e)
        task.save(update_fields=['status', 'error_message'])


def dispatch_evolution(session):
    if settings.USE_CELERY:
        run_evolution_task.delay(str(session.id))
    else:
        run_evolution_task(str(session.id))
