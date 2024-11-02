from django.apps import AppConfig
import sys

class SubscriptionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'subscription'

    def ready(self):
        if 'runserver' in sys.argv or 'shell' in sys.argv:
            from .subscription_checker import SubscriptionChecker
            from .utils import rate_limiter

            # 初始化 SubscriptionChecker
            self.subscription_checker = SubscriptionChecker()
            self.rate_limiter = rate_limiter

            print("SubscriptionChecker 已启动。")
