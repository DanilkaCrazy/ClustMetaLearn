import json
import os
import shutil
from collections import Counter
from datetime import datetime

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.core.files.storage import FileSystemStorage
from django.db.models import Avg, Count
from django.db.models.functions import TruncDate
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone, translation
from django.utils.translation import gettext as _
from django.views.decorators.http import require_GET, require_POST

try:
    from django.utils.translation import LANGUAGE_SESSION_KEY
except ImportError:
    LANGUAGE_SESSION_KEY = settings.LANGUAGE_COOKIE_NAME

from .charts import (
    ari_scatter_chart, dashboard_daily_chart, evolution_placeholder_chart,
    meta_correlation_heatmap, meta_histogram_chart, pca_scatter_chart, top3_bar_chart,
)
from .dataset_import import (
    cleanup_temp_dir, download_hf_dataset, download_kaggle_dataset,
    parse_hf_ref, parse_kaggle_ref,
)
from .forms import (
    DatasetUploadForm, EvolutionForm, HuggingFaceImportForm,
    KaggleImportForm, ProfileForm, RegisterForm,
)
from .models import ClusteringTask, EvolutionarySession, UserProfile
from .reports import (
    MAX_EXPORT_TASKS, ari_trend_png, build_all_report_text,
    build_pdf_from_html, build_single_report_text, render_report_html,
)
from .tasks import dispatch_evolution, run_async_pipeline
from .utils import (
    extract_meta_features, format_algorithm, get_top3_probabilities,
    predict_clustering_strategy,
)


def _get_or_create_profile(user):
    profile, created = UserProfile.objects.get_or_create(user=user)
    return profile


def _apply_user_preferences(request):
    if request.user.is_authenticated:
        profile = _get_or_create_profile(request.user)
        request.session['theme'] = profile.theme_preference
        translation.activate(profile.language)
        request.session[LANGUAGE_SESSION_KEY] = profile.language
        request.session[settings.LANGUAGE_COOKIE_NAME] = profile.language


def _save_task_file(task, src_path, original_name):
    datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    dest_path = os.path.join(datasets_dir, f'task_{task.id}_{original_name}')
    if os.path.abspath(src_path) != os.path.abspath(dest_path):
        shutil.copy2(src_path, dest_path)
    return dest_path


def _process_upload(request, task, file_path):
    if settings.USE_CELERY:
        task.status = 'processing'
        task.save(update_fields=['status'])
        run_async_pipeline.delay(str(task.id), file_path)
        messages.success(request, _('Dataset uploaded. Analysis is running in background.'))
        return redirect('clustering:recommend', task_id=task.id)

    meta = extract_meta_features(
        file_path, fmt=task.input_format,
        n_samples=task.n_samples, n_features=task.n_features,
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
    messages.success(request, _('Dataset analyzed successfully'))
    return redirect('clustering:recommend', task_id=task.id)


def _create_task_from_file(request, user, file_path, file_name, input_format='csv',
                           n_samples=None, n_features=None, data_source='file',
                           source_identifier=''):
    task = ClusteringTask.objects.create(
        user=user,
        file_name=file_name,
        data_source=data_source,
        source_identifier=source_identifier,
        input_format=input_format,
        n_samples=n_samples,
        n_features=n_features,
        status='processing',
    )
    try:
        final_path = _save_task_file(task, file_path, file_name)
        return _process_upload(request, task, final_path)
    except Exception as e:
        task.status = 'failed'
        task.error_message = str(e)
        task.save()
        messages.error(request, _('Analysis failed: %(error)s') % {'error': str(e)})
        return None


def index(request):
    return render(request, 'clustering/index.html', {'hide_sidebar': True})


def demo_redirect(request):
    if request.user.is_authenticated:
        return redirect('clustering:dashboard')
    return redirect('clustering:login')


def login_view(request):
    if request.user.is_authenticated:
        return redirect('clustering:dashboard')
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            _apply_user_preferences(request)
            theme = request.POST.get('sync_theme')
            if theme in ('light', 'dark'):
                profile = _get_or_create_profile(user)
                profile.theme_preference = theme
                profile.save(update_fields=['theme_preference'])
                request.session['theme'] = theme
            messages.success(request, _('Welcome back!'))
            return redirect('clustering:dashboard')
        messages.error(request, _('Invalid username or password'))
    else:
        form = AuthenticationForm()
    return render(request, 'clustering/login.html', {'form': form})


def register_view(request):
    if request.user.is_authenticated:
        return redirect('clustering:dashboard')
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            theme = request.POST.get('sync_theme', 'light')
            lang = request.POST.get('sync_language', 'en')
            UserProfile.objects.create(
                user=user,
                theme_preference=theme if theme in ('light', 'dark') else 'light',
                language=lang if lang in ('en', 'ru') else 'en',
            )
            login(request, user)
            _apply_user_preferences(request)
            messages.success(request, _('Account created successfully'))
            return redirect('clustering:dashboard')
        messages.error(request, _('Please correct the errors below'))
    else:
        form = RegisterForm()
    return render(request, 'clustering/register.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('clustering:index')


@login_required
def dashboard(request):
    tasks_qs = ClusteringTask.objects.filter(user=request.user)
    success_tasks = tasks_qs.filter(status='success')
    failed_tasks = tasks_qs.filter(status='failed')
    stats = {
        'total': tasks_qs.count(),
        'avg_ari': success_tasks.aggregate(v=Avg('ari_prediction'))['v'],
        'success_count': success_tasks.count(),
        'failed_count': failed_tasks.count(),
    }
    algo_counts = Counter(
        success_tasks.values_list('recommended_algorithm', flat=True)
    )
    stats['top_algorithms'] = algo_counts.most_common(3)
    daily = (
        tasks_qs.annotate(day=TruncDate('created_at'))
        .values('day').annotate(count=Count('id')).order_by('day')
    )
    daily_counts = {str(d['day']): d['count'] for d in daily if d['day']}
    last_tasks = tasks_qs.order_by('-created_at')[:5]
    chart_daily = None
    if daily_counts:
        is_dark = request.session.get('theme', 'light') == 'dark'
        if request.user.is_authenticated:
            try:
                is_dark = request.user.profile.theme_preference == 'dark'
            except Exception:
                pass
        chart_daily = json.loads(dashboard_daily_chart(daily_counts, dark=is_dark))
    return render(request, 'clustering/dashboard.html', {
        'tasks': tasks_qs.order_by('-created_at'),
        'stats': stats,
        'last_tasks': last_tasks,
        'chart_daily_json': chart_daily,
    })


@login_required
def upload_view(request):
    file_form = DatasetUploadForm()
    kaggle_form = KaggleImportForm()
    hf_form = HuggingFaceImportForm()
    active_tab = 'file'

    if request.method == 'POST':
        source = request.POST.get('source', 'file')
        active_tab = source

        if source == 'file':
            file_form = DatasetUploadForm(request.POST, request.FILES)
            if file_form.is_valid():
                uploaded = file_form.cleaned_data['dataset_file']
                ext = uploaded.name.rsplit('.', 1)[-1].lower()
                task = ClusteringTask.objects.create(
                    user=request.user,
                    file_name=uploaded.name,
                    data_source='file',
                    input_format='bin' if ext == 'bin' else 'csv',
                    n_samples=file_form.cleaned_data.get('n_samples'),
                    n_features=file_form.cleaned_data.get('n_features'),
                    status='processing',
                )
                fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'datasets'))
                filename = fs.save(f'task_{task.id}_{task.file_name}', uploaded)
                file_path = fs.path(filename)
                try:
                    return _process_upload(request, task, file_path)
                except Exception as e:
                    task.status = 'failed'
                    task.error_message = str(e)
                    task.save()
                    messages.error(request, _('Analysis failed: %(error)s') % {'error': str(e)})
            else:
                messages.error(request, _('Please correct the form errors'))

        elif source == 'kaggle':
            kaggle_form = KaggleImportForm(request.POST)
            if kaggle_form.is_valid():
                ref = kaggle_form.cleaned_data['kaggle_ref']
                temp_root = None
                try:
                    slug, display = parse_kaggle_ref(ref)
                    messages.info(request, _('Downloading from Kaggle: %(name)s…') % {'name': display})
                    csv_path, display, temp_root = download_kaggle_dataset(ref)
                    file_name = f'kaggle_{display.replace("/", "_")}.csv'
                    result = _create_task_from_file(
                        request, request.user, csv_path, file_name,
                        data_source='kaggle', source_identifier=slug,
                    )
                    if result:
                        return result
                except Exception as e:
                    messages.error(request, _('Kaggle import failed: %(error)s') % {'error': str(e)})
                finally:
                    cleanup_temp_dir(temp_root)
            else:
                messages.error(request, _('Please correct the form errors'))

        elif source == 'huggingface':
            hf_form = HuggingFaceImportForm(request.POST)
            if hf_form.is_valid():
                ref = hf_form.cleaned_data['hf_ref']
                temp_root = None
                try:
                    slug, display = parse_hf_ref(ref)
                    messages.info(request, _('Downloading from Hugging Face: %(name)s…') % {'name': display})
                    csv_path, display, temp_root = download_hf_dataset(ref)
                    file_name = f'hf_{display.replace("/", "_")}.csv'
                    result = _create_task_from_file(
                        request, request.user, csv_path, file_name,
                        data_source='huggingface', source_identifier=slug,
                    )
                    if result:
                        return result
                except Exception as e:
                    messages.error(request, _('Hugging Face import failed: %(error)s') % {'error': str(e)})
                finally:
                    cleanup_temp_dir(temp_root)
            else:
                messages.error(request, _('Please correct the form errors'))

    return render(request, 'clustering/upload.html', {
        'form': file_form,
        'kaggle_form': kaggle_form,
        'hf_form': hf_form,
        'active_tab': active_tab,
    })


@login_required
def recommend_view(request, task_id):
    task = get_object_or_404(ClusteringTask, id=task_id, user=request.user)
    meta = task.meta_features_json or {}
    top3_probs = get_top3_probabilities(meta) if meta else {}
    hp = task.hp_intervals or {}
    hp_display = {
        'k_min': hp.get('k_min', 'N/A'),
        'k_max': hp.get('k_max', 'N/A'),
        'k_median': hp.get('k_median', 'N/A'),
        'scaler_rate': hp.get('scaler_rate', 'N/A'),
    }
    return render(request, 'clustering/recommend.html', {
        'task': task,
        'meta_features': meta,
        'top3_display': [format_algorithm(a) for a in (task.top3_algorithms or [])],
        'top3_probs': top3_probs,
        'top3_probs_json': json.dumps(top3_probs),
        'algo_display': format_algorithm(task.recommended_algorithm) if task.recommended_algorithm else 'N/A',
        'hp_display': hp_display,
        'hp_json': json.dumps(hp_display),
    })


@login_required
@require_GET
def task_meta_api(request, task_id):
    task = get_object_or_404(ClusteringTask, id=task_id, user=request.user)
    meta = task.meta_features_json or {}
    return JsonResponse({
        'status': task.status,
        'meta_features': meta,
        'recommended_metric': task.recommended_metric or 'N/A',
        'recommended_algorithm': format_algorithm(task.recommended_algorithm) if task.recommended_algorithm else 'N/A',
        'ari_prediction': task.ari_prediction,
        'top3': [format_algorithm(a) for a in (task.top3_algorithms or [])],
        'hp_intervals': task.hp_intervals or {},
    })


@login_required
def evolve_view(request, task_id):
    task = get_object_or_404(ClusteringTask, id=task_id, user=request.user)
    session, created = EvolutionarySession.objects.get_or_create(
        task=task,
        defaults={
            'total_generations': 10,
            'population_size': 40,
            'fitness_metric': task.recommended_metric or 'silhouette',
        },
    )
    if request.method == 'POST':
        action = request.POST.get('action', 'save')
        form = EvolutionForm(request.POST)
        if form.is_valid():
            session.total_generations = form.cleaned_data['total_generations']
            session.population_size = form.cleaned_data['population_size']
            session.fitness_metric = form.cleaned_data['fitness_metric']
            session.bandit_strategy = form.cleaned_data['bandit_strategy']
            session.save()
            if action == 'run':
                session.status = 'running'
                session.current_generation = 0
                session.generations.all().delete()
                session.save()
                dispatch_evolution(session)
                messages.info(request, _('Evolution started. Track progress below.'))
                return redirect('clustering:evolve_status', task_id=task.id)
            messages.success(request, _('Evolution parameters saved.'))
            return redirect('clustering:evolve', task_id=task.id)
        messages.error(request, _('Please correct the form errors'))
    else:
        form = EvolutionForm(initial={
            'total_generations': session.total_generations,
            'population_size': session.population_size,
            'fitness_metric': session.fitness_metric,
            'bandit_strategy': session.bandit_strategy,
        })
    gens = list(session.generations.order_by('generation_number').values(
        'generation_number', 'fitness_score',
    ))
    chart_evolution = None
    if gens:
        chart_evolution = json.loads(evolution_placeholder_chart([
            {'generation': g['generation_number'], 'fitness': g['fitness_score']} for g in gens
        ]))
    return render(request, 'clustering/evolve.html', {
        'task': task, 'session': session, 'form': form,
        'algo_display': format_algorithm(task.recommended_algorithm),
        'chart_evolution': chart_evolution,
    })


@login_required
def evolve_status_view(request, task_id):
    task = get_object_or_404(ClusteringTask, id=task_id, user=request.user)
    session = get_object_or_404(EvolutionarySession, task=task)
    progress = 0
    if session.total_generations:
        progress = int(100 * session.current_generation / session.total_generations)
    return render(request, 'clustering/evolve_status.html', {
        'task': task, 'session': session, 'progress': progress,
    })


@login_required
@require_GET
def evolve_status_api(request, task_id):
    task = get_object_or_404(ClusteringTask, id=task_id, user=request.user)
    session = get_object_or_404(EvolutionarySession, task=task)
    progress = 0
    if session.total_generations:
        progress = int(100 * session.current_generation / session.total_generations)
    gens = list(session.generations.order_by('generation_number').values(
        'generation_number', 'fitness_score', 'best_pipeline',
    ))
    return JsonResponse({
        'status': session.status,
        'current_generation': session.current_generation,
        'total_generations': session.total_generations,
        'progress': progress,
        'best_fitness': session.best_fitness,
        'best_pipeline': session.best_pipeline,
        'generations': gens,
    })


@login_required
def analyze_view(request, task_id):
    task = get_object_or_404(ClusteringTask, id=task_id, user=request.user)
    meta = task.meta_features_json or {}
    user_tasks = ClusteringTask.objects.filter(
        user=request.user, status='success',
    ).exclude(ari_prediction__isnull=True).order_by('created_at')
    tasks_data = [
        {'name': t.file_name, 'ari': t.ari_prediction or 0}
        for t in user_tasks
    ]
    all_metas = [
        t.meta_features_json for t in user_tasks if t.meta_features_json
    ]
    top3_probs = get_top3_probabilities(meta) if meta else {}
    is_dark = request.session.get('theme', 'light') == 'dark'
    if request.user.is_authenticated:
        try:
            is_dark = request.user.profile.theme_preference == 'dark'
        except Exception:
            pass
    charts = {}
    if meta:
        charts['histogram'] = json.loads(meta_histogram_chart(meta, dark=is_dark))
        charts['correlation'] = json.loads(meta_correlation_heatmap(meta, all_metas, dark=is_dark))
        charts['pca'] = json.loads(pca_scatter_chart(meta, task.recommended_algorithm or '', dark=is_dark))
        charts['top3'] = json.loads(top3_bar_chart(top3_probs, dark=is_dark))
    charts['ari_scatter'] = json.loads(ari_scatter_chart(tasks_data, dark=is_dark))
    session = EvolutionarySession.objects.filter(task=task).first()
    gens = []
    if session:
        gens = list(session.generations.order_by('generation_number').values(
            'generation_number', 'fitness_score',
        ))
    charts['evolution'] = json.loads(evolution_placeholder_chart(
        [{'generation': g['generation_number'], 'fitness': g['fitness_score']} for g in gens]
        if gens else None,
        dark=is_dark,
    ))
    return render(request, 'clustering/analyze.html', {
        'task': task,
        'meta_features': meta,
        'charts_data': charts,
        'top3_probs': top3_probs,
    })


@login_required
def profile_view(request):
    profile = _get_or_create_profile(request.user)
    if request.method == 'POST':
        form = ProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            request.session['theme'] = profile.theme_preference
            translation.activate(profile.language)
            request.session[LANGUAGE_SESSION_KEY] = profile.language
            request.session[settings.LANGUAGE_COOKIE_NAME] = profile.language
            messages.success(request, _('Profile updated'))
            return redirect('clustering:profile')
    else:
        form = ProfileForm(instance=profile)
    return render(request, 'clustering/profile.html', {'form': form, 'profile': profile})


def about_view(request):
    return render(request, 'clustering/about.html', {'hide_sidebar': not request.user.is_authenticated})


def docs_view(request):
    return render(request, 'clustering/docs.html', {'hide_sidebar': not request.user.is_authenticated})


def privacy_view(request):
    return render(request, 'clustering/privacy.html', {'hide_sidebar': not request.user.is_authenticated})


@require_POST
def set_theme(request):
    theme = request.POST.get('theme', 'light')
    if theme not in ('light', 'dark'):
        theme = 'light'
    request.session['theme'] = theme
    if request.user.is_authenticated:
        profile = _get_or_create_profile(request.user)
        profile.theme_preference = theme
        profile.save(update_fields=['theme_preference'])
    referer = request.META.get('HTTP_REFERER')
    return redirect(referer or 'clustering:index')


@require_POST
def set_language(request):
    lang = request.POST.get('language', 'en')
    if lang not in ('en', 'ru'):
        lang = 'en'
    translation.activate(lang)
    request.session[LANGUAGE_SESSION_KEY] = lang
    request.session[settings.LANGUAGE_COOKIE_NAME] = lang
    if request.user.is_authenticated:
        profile = _get_or_create_profile(request.user)
        profile.language = lang
        profile.save(update_fields=['language'])
    referer = request.META.get('HTTP_REFERER')
    response = redirect(referer or 'clustering:index')
    response.set_cookie(settings.LANGUAGE_COOKIE_NAME, lang)
    return response


@login_required
@require_GET
def set_language_api(request):
    """JSON language switch for AJAX without full page semantics."""
    lang = request.GET.get('language', 'en')
    if lang not in ('en', 'ru'):
        return JsonResponse({'error': 'invalid language'}, status=400)
    translation.activate(lang)
    request.session[LANGUAGE_SESSION_KEY] = lang
    request.session[settings.LANGUAGE_COOKIE_NAME] = lang
    if request.user.is_authenticated:
        profile = _get_or_create_profile(request.user)
        profile.language = lang
        profile.save(update_fields=['language'])
    response = JsonResponse({'language': lang, 'reload': True})
    response.set_cookie(settings.LANGUAGE_COOKIE_NAME, lang)
    return response


def _export_filename(prefix, username, ext, task_id=None):
    date_str = datetime.now().strftime('%Y%m%d')
    if task_id:
        return f'report_experiment_{task_id}_{date_str}.{ext}'
    return f'report_all_experiments_{username}_{date_str}.{ext}'


@login_required
@require_GET
def export_txt_view(request, task_id=None):
    scope = request.GET.get('scope', 'single')
    if task_id:
        task = get_object_or_404(ClusteringTask, id=task_id, user=request.user)
        content = build_single_report_text(task)
        filename = _export_filename('experiment', request.user.username, 'txt', task_id=task_id)
    elif scope == 'all':
        tasks = ClusteringTask.objects.filter(user=request.user).order_by('-created_at')
        count = tasks.count()
        if count > MAX_EXPORT_TASKS:
            messages.warning(
                request,
                _('Report limited to %(max)s experiments (you have %(n)s).') % {
                    'max': MAX_EXPORT_TASKS, 'n': count,
                },
            )
            tasks = tasks[:MAX_EXPORT_TASKS]
        try:
            content = build_all_report_text(tasks)
        except ValueError as e:
            messages.error(request, str(e))
            return redirect('clustering:dashboard')
        filename = _export_filename('all', request.user.username, 'txt')
    else:
        messages.error(request, _('Invalid export scope'))
        return redirect('clustering:dashboard')

    response = HttpResponse(content, content_type='text/plain; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


@login_required
@require_GET
def export_pdf_view(request, task_id=None):
    scope = request.GET.get('scope', 'single')
    chart_png = None

    if task_id:
        task = get_object_or_404(ClusteringTask, id=task_id, user=request.user)
        text = build_single_report_text(task)
        title = _('Experiment Report')
        filename = _export_filename('experiment', request.user.username, 'pdf', task_id=task_id)
    elif scope == 'all':
        tasks = list(
            ClusteringTask.objects.filter(user=request.user).order_by('-created_at')
        )
        if len(tasks) > MAX_EXPORT_TASKS:
            messages.warning(
                request,
                _('Report limited to %(max)s experiments (you have %(n)s).') % {
                    'max': MAX_EXPORT_TASKS, 'n': len(tasks),
                },
            )
            tasks = tasks[:MAX_EXPORT_TASKS]
        try:
            text = build_all_report_text(tasks)
        except ValueError as e:
            messages.error(request, str(e))
            return redirect('clustering:dashboard')
        title = _('All Experiments Report')
        filename = _export_filename('all', request.user.username, 'pdf')
        chart_png = ari_trend_png(tasks)
    else:
        messages.error(request, _('Invalid export scope'))
        return redirect('clustering:dashboard')

    html = render_report_html(title, [text], chart_png=chart_png)
    try:
        pdf_bytes = build_pdf_from_html(html)
    except ImportError:
        messages.error(request, _('PDF export requires weasyprint. Install it from requirements.txt.'))
        return redirect(request.META.get('HTTP_REFERER', 'clustering:dashboard'))

    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response
