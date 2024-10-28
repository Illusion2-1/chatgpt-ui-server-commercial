from rest_framework import serializers
from .models import RedeemCode, Subscription

class RedeemCodeSerializer(serializers.ModelSerializer):
    class Meta:
        model = RedeemCode
        fields = ['id', 'code', 'amount', 'created_at', 'is_used', 'used_by', 'used_at']

class SubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subscription
        fields = ['id', 'title', 'available_models', 'price', 'duration', 'rate_limit', 'device_limit']
