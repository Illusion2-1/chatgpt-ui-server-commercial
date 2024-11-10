from django.contrib import admin
from .models import Announcement, AnnouncementSetting

@admin.register(Announcement)
class AnnouncementAdmin(admin.ModelAdmin):
    list_display = ('title', 'created_at')
    search_fields = ('title', 'content')

@admin.register(AnnouncementSetting)
class AnnouncementSettingAdmin(admin.ModelAdmin):
    list_display = ('announcement', 'display_until', 'is_active')
    list_filter = ('is_active',)
