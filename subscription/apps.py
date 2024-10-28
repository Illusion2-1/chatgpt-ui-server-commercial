from django.apps import AppConfig
import sys
#from .utils import rate_limiter

class SubscriptionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'subscription'

    def ready(self):
        if 'runserver' in sys.argv or 'shell' in sys.argv:
            from django.db.models.signals import post_migrate
            from django.dispatch import receiver

            @receiver(post_migrate)
            def initialize_app(sender, **kwargs):
                if sender.name == self.name:
                    from .subscription_checker import SubscriptionChecker
                    from .utils import rate_limiter
                    self.subscription_checker = SubscriptionChecker()
                    self.rate_limiter = rate_limiter
