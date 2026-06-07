import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('clustering', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='clusteringtask',
            name='user',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='tasks',
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name='clusteringtask',
            name='top3_algorithms',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='clusteringtask',
            name='hp_intervals',
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='clusteringtask',
            name='predicted_cvi_metric',
            field=models.CharField(blank=True, max_length=100),
        ),
        migrations.RenameField(
            model_name='clusteringtask',
            old_name='predicted_ari',
            new_name='ari_prediction',
        ),
        migrations.AddField(
            model_name='evolutionarysession',
            name='task',
            field=models.OneToOneField(
                blank=True, null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='evolution',
                to='clustering.clusteringtask',
            ),
        ),
        migrations.RemoveField(
            model_name='evolutionarysession',
            name='analysis',
        ),
    ]
