from django.db import models
from django.contrib.auth import get_user_model
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone
import uuid

User = get_user_model()

class RedeemCode(models.Model):
    code = models.CharField(max_length=100, unique=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    is_used = models.BooleanField(default=False)
    used_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    used_at = models.DateTimeField(null=True, blank=True)

class Subscription(models.Model):
    title = models.CharField(max_length=100)
    available_models = models.JSONField()  # 存储可用模型列表
    price = models.DecimalField(max_digits=10, decimal_places=2)
    duration = models.DurationField()  # 订阅时长
    rate_limit = models.IntegerField()  # 速率限制
    device_limit = models.IntegerField()  # 同时在线设备限制

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    balance = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    subscription = models.ForeignKey('subscription.Subscription', null=True, blank=True, on_delete=models.SET_NULL)
    subscription_expiry = models.DateTimeField(null=True, blank=True)
    subscription_is_active = models.BooleanField(default=False)

    def update_subscription_status(self):
        from django.utils import timezone
        if self.subscription_expiry and self.subscription_expiry <= timezone.now():
            self.subscription_is_active = False
            self.save()

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()