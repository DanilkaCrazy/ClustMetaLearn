from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.utils.translation import gettext_lazy as _

from .models import ClusteringTask, UserProfile


class DatasetUploadForm(forms.ModelForm):
    dataset_file = forms.FileField(
        label=_('Dataset file (.csv, .bin)'),
        required=False,
        widget=forms.ClearableFileInput(attrs={
            'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface',
            'accept': '.csv,.bin',
        }),
    )

    class Meta:
        model = ClusteringTask
        fields = ['input_format', 'n_samples', 'n_features']
        widgets = {
            'input_format': forms.Select(
                choices=[('csv', _('CSV Table')), ('bin', _('Binary (.bin)'))],
                attrs={'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface'},
            ),
            'n_samples': forms.NumberInput(attrs={
                'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface',
                'placeholder': _('Required for .bin'),
            }),
            'n_features': forms.NumberInput(attrs={
                'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface',
                'placeholder': _('Required for .bin'),
            }),
        }

    def clean_dataset_file(self):
        f = self.cleaned_data.get('dataset_file')
        if not f:
            return f
        ext = f.name.rsplit('.', 1)[-1].lower()
        if ext not in ('csv', 'bin'):
            raise forms.ValidationError(_('Only .csv and .bin files are supported'))
        return f

    def clean(self):
        cleaned = super().clean()
        f = cleaned.get('dataset_file')
        if not f:
            raise forms.ValidationError(_('Please select a dataset file'))
        fmt = cleaned.get('input_format')
        if f:
            ext = f.name.rsplit('.', 1)[-1].lower()
            if ext == 'bin':
                cleaned['input_format'] = 'bin'
                if not cleaned.get('n_samples') or not cleaned.get('n_features'):
                    raise forms.ValidationError(
                        _('n_samples and n_features are required for .bin files')
                    )
            elif ext == 'csv':
                cleaned['input_format'] = 'csv'
        elif fmt == 'bin' and (not cleaned.get('n_samples') or not cleaned.get('n_features')):
            raise forms.ValidationError(_('n_samples and n_features are required for .bin files'))
        return cleaned


class KaggleImportForm(forms.Form):
    kaggle_ref = forms.CharField(
        label=_('Kaggle dataset (name or URL)'),
        widget=forms.TextInput(attrs={
            'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface',
            'placeholder': 'uciml/iris or https://www.kaggle.com/datasets/uciml/iris',
            'id': 'kaggle-ref-input',
        }),
        help_text=_('Example: uciml/iris or full Kaggle URL'),
    )


class HuggingFaceImportForm(forms.Form):
    hf_ref = forms.CharField(
        label=_('Hugging Face dataset (id or URL)'),
        widget=forms.TextInput(attrs={
            'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface',
            'placeholder': 'scikit-learn/iris or https://huggingface.co/datasets/scikit-learn/iris',
            'id': 'hf-ref-input',
        }),
        help_text=_('Example: scikit-learn/iris or full Hugging Face URL'),
    )


class ProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['avatar', 'phone_number', 'theme_preference', 'company_org', 'language']
        widgets = {
            'phone_number': forms.TextInput(attrs={
                'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface',
            }),
            'company_org': forms.TextInput(attrs={
                'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface',
            }),
            'theme_preference': forms.RadioSelect(choices=[
                ('light', _('Light')), ('dark', _('Dark')),
            ]),
            'language': forms.Select(choices=[
                ('en', 'English'), ('ru', 'Русский'),
            ], attrs={'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface'}),
        }


class EvolutionForm(forms.Form):
    total_generations = forms.IntegerField(
        label=_('Total generations'), initial=10, min_value=1, max_value=100,
        widget=forms.NumberInput(attrs={'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface'}),
    )
    population_size = forms.IntegerField(
        label=_('Population size'), initial=40, min_value=10, max_value=200,
        widget=forms.NumberInput(attrs={'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface'}),
    )
    fitness_metric = forms.ChoiceField(
        label=_('Fitness metric'),
        choices=[
            ('silhouette', _('Silhouette')),
            ('calinski_harabasz', _('Calinski-Harabasz')),
            ('davies_bouldin', _('Davies-Bouldin')),
        ],
        widget=forms.Select(attrs={'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface'}),
    )
    bandit_strategy = forms.ChoiceField(
        label=_('Bandit strategy'),
        choices=[
            ('ucb1', 'UCB1'), ('thompson', 'Thompson'), ('epsilon', _('Epsilon-Greedy')),
        ],
        widget=forms.Select(attrs={'class': 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface'}),
    )


class RegisterForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        fields = ('username', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'w-full rounded-md border border-outline-variant p-2 bg-surface-container-lowest text-on-surface'
