import threading
import time
from django.utils import timezone
from django.db import transaction
from .models import Profile

class SubscriptionChecker:
    def __init__(self, check_interval=60):
        """
        初始化订阅检查器。

        :param check_interval: 检查间隔时间（秒）
        """
        self.check_interval = check_interval
        self.subscriptions = {}
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        print("初始化订阅检查器...")
        print(f"检查间隔时间: {check_interval}秒")
        # 在启动线程前进行一次全面检查
        self.check_all_subscriptions()
        
        # 加载剩余的有效订阅
        self.load_subscriptions()
        
        self.thread.start()

    def check_all_subscriptions(self):
        """
        检查所有订阅，将过期的订阅设置为不活跃。
        """
        now = timezone.now()
        expired_profiles = Profile.objects.filter(
            subscription_is_active=True,
            subscription_expiry__lte=now
        )
        self.deactivate_subscriptions(expired_profiles)

    def load_subscriptions(self):
        """
        加载所有有效的订阅到内存字典中。
        """
        active_profiles = Profile.objects.filter(subscription_is_active=True, subscription_expiry__gt=timezone.now())
        with self.lock:
            for profile in active_profiles:
                self.subscriptions[profile.user_id] = profile.subscription_expiry

    def run(self):
        """
        后台线程运行的方法，定期检查订阅是否到期。
        """
        while self.running:
            now = timezone.now()
            expired_users = []
            with self.lock:
                for user_id, expiry in list(self.subscriptions.items()):
                    if expiry <= now:
                        expired_users.append(user_id)
                        del self.subscriptions[user_id]

            if expired_users:
                self.deactivate_subscriptions(expired_users)

            time.sleep(self.check_interval)

    @transaction.atomic
    def deactivate_subscriptions(self, profiles):
        """
        将到期的订阅设为不活跃。

        :param profiles: 到期的用户Profile对象列表
        """
        for profile in profiles:
            profile.subscription_is_active = False
            profile.save()

    def stop(self):
        """
        停止后台线程。
        """
        self.running = False
        self.thread.join()
