from django.db import models
from rest_framework import serializers
from .models import Conversation, Message, Prompt, EmbeddingDocument, Setting, LanguageModel

class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ['id', 'topic', 'created_at']

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'message', 'is_bot', 'message_type', 'embedding_message_doc', 'created_at', 'image_hash']


class PromptSerializer(serializers.ModelSerializer):

    prompt = serializers.CharField(trim_whitespace=False, allow_blank=True)

    class Meta:
        model = Prompt
        fields = ['id', 'title', 'prompt', 'created_at', 'updated_at']


class EmbeddingDocumentSerializer(serializers.ModelSerializer):
    '''embedding document store'''
    class Meta:
        ''' select fields'''
        model = EmbeddingDocument
        fields = ['id', 'title', 'created_at']
        read_only_fields = ('faiss_store', 'created_at')


class SettingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Setting
        fields = ('name', 'value')

class LanguageModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = LanguageModel
        fields = ['id', 'name', 'display_name', 'max_tokens', 'max_prompt_tokens', 'max_response_tokens', 'created_at', 'updated_at', 'image_support']