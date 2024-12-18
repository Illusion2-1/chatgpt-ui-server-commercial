# Generated by Django 4.1.7 on 2024-11-10 10:08

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Announcement',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200, verbose_name='标题')),
                ('content', models.TextField(verbose_name='内容')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='创建时间')),
            ],
        ),
        migrations.CreateModel(
            name='AnnouncementSetting',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('display_until', models.DateTimeField(verbose_name='展示截至日期')),
                ('is_active', models.BooleanField(default=True, verbose_name='是否激活')),
                ('announcement', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='setting', to='announcements.announcement', verbose_name='公告')),
            ],
        ),
    ]
