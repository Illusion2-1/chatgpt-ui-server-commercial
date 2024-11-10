from django.apps import AppConfig
import sys
import os

class SubscriptionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'subscription'
    initialized = False  # 添加标志变量

    def ready(self):
        if os.environ.get('RUN_MAIN') == 'true':
            if 'runserver' in sys.argv or 'shell' in sys.argv:
                from .subscription_checker import SubscriptionChecker
                from .utils import rate_limiter
                if not self.initialized:
                    self.subscription_checker = SubscriptionChecker()
                    self.rate_limiter = rate_limiter
                    self.initialized = True

