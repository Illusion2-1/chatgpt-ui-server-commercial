from django.contrib import admin

from .models import Conversation, Message, Setting, LanguageModel


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'topic', 'created_at')
    list_editable = ('topic',)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'get_conversation_topic', 'message', 'is_bot', 'tokens','created_at')
    list_editable = ('message', 'is_bot', 'tokens')

    def get_conversation_topic(self, obj):
        return obj.conversation.topic

    get_conversation_topic.short_description = 'Conversation Topic'


@admin.register(Setting)
class SettingAdmin(admin.ModelAdmin):
    list_display = ('name', 'value')
    list_editable = ('value',)


@admin.register(LanguageModel)
class LanguageModelAdmin(admin.ModelAdmin):
    search_fields = ('name', 'display_name')