# Generated manually for ClustMetaLearn

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('clustering', '0003_alter_clusteringtask_status_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='clusteringtask',
            name='data_source',
            field=models.CharField(
                choices=[
                    ('file', 'File upload'),
                    ('kaggle', 'Kaggle'),
                    ('huggingface', 'Hugging Face'),
                ],
                default='file',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='clusteringtask',
            name='source_identifier',
            field=models.CharField(blank=True, max_length=500),
        ),
    ]
