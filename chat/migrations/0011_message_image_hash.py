# Generated by Django 4.1.7 on 2024-11-17 15:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0010_languagemodel_image_support'),
    ]

    operations = [
        migrations.AddField(
            model_name='message',
            name='image_hash',
            field=models.TextField(default=''),
        ),
    ]
