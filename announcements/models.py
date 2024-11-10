from django.db import models

# Create your models here.

class Announcement(models.Model):
    title = models.CharField(max_length=200, verbose_name="标题")
    content = models.TextField(verbose_name="内容")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    def __str__(self):
        return self.title

class AnnouncementSetting(models.Model):
    announcement = models.OneToOneField(Announcement, on_delete=models.CASCADE, related_name='setting', verbose_name="公告")
    display_until = models.DateTimeField(verbose_name="展示截至日期")
    is_active = models.BooleanField(default=True, verbose_name="是否激活")

    def __str__(self):
        return f"设置 - {self.announcement.title}"
