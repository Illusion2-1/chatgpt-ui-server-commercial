from django.contrib.sessions.models import Session
from django.utils import timezone
from django.conf import settings
import json
import threading
import time
from .models import Profile

def get_user_sessions(user):
    sessions = Session.objects.filter(expire_date__gte=timezone.now())
    user_sessions = []
    for session in sessions:
        data = session.get_decoded()
        if data.get('_auth_user_id') == str(user.id):
            user_sessions.append(session)
    return user_sessions

class RateLimiter:
    """
    RateLimiter类用于实现用户请求的速率限制功能。
    它通过记录每个用户的请求次数并与其订阅的速率限制进行比较来实现。
    """

    def __init__(self):
        # 用于存储用户使用情况的字典，键为用户ID，值为元组(时间窗口, 请求次数)
        self.usage_cache = {}
        # 线程锁，用于确保在多线程环境下对usage_cache的操作是线程安全的
        self.lock = threading.Lock()
        # 启动后台线程定期清理缓存
        cleaner_thread = threading.Thread(target=self.clean_cache, daemon=True)
        cleaner_thread.start()

    def record_usage(self, user_id):
        """
        记录用户的使用情况。
        每次用户发送请求时调用此方法来增加其使用计数。

        :param user_id: 用户的唯一标识符
        """
        now = timezone.now()
        current_minute = now.replace(second=0, microsecond=0)
        with self.lock:
            user_usage = self.usage_cache.get(user_id)
            if user_usage:
                window, count = user_usage
                if window == current_minute:
                    # 如果是同一分钟内的请求，增加计数
                    self.usage_cache[user_id] = (window, count + 1)
                else:
                    # 如果是新的一分钟，重置计数
                    self.usage_cache[user_id] = (current_minute, 1)
            else:
                # 如果是该用户的第一次请求，初始化计数
                self.usage_cache[user_id] = (current_minute, 1)

    def is_rate_limited(self, user_id):
        """
        检查用户是否超过了速率限制。

        :param user_id: 用户的唯一标识符
        :return: 如果用户超过速率限制返回True，否则返回False
        """
        now = timezone.now()
        current_minute = now.replace(second=0, microsecond=0)
        with self.lock:
            user_usage = self.usage_cache.get(user_id)
            if not user_usage:
                # 如果用户不在缓存中，初始化计数
                self.usage_cache[user_id] = (current_minute, 1)
                user_usage = (current_minute, 1)

            window, count = user_usage
            if window != current_minute:
                # 如果是新的一分钟，重置计数
                count = 1
                self.usage_cache[user_id] = (current_minute, count)

            try:
                profile = Profile.objects.get(user_id=user_id)
                if profile.subscription and profile.subscription_is_active:
                    rate_limit = profile.subscription.rate_limit
                else:
                    rate_limit = 0 
            except Profile.DoesNotExist:
                rate_limit = 0

            # 检查是否超过速率限制
            is_limited = count > rate_limit

            # 增加计数
            self.usage_cache[user_id] = (current_minute, count + 1)

            return is_limited

    def clean_cache(self):
        """
        定期清理缓存中超过一定时间的记录。
        这个方法在一个后台线程中运行，每5分钟清理一次缓存。
        """
        while True:
            now = timezone.now()
            cutoff = now - timezone.timedelta(minutes=2)
            with self.lock:
                # 删除超过2分钟的记录
                keys_to_delete = [user_id for user_id, (window, _) in self.usage_cache.items() if window < cutoff]
                for user_id in keys_to_delete:
                    del self.usage_cache[user_id]
            time.sleep(60)  # 每分钟清理一次

# 创建RateLimiter的单例实例以供全局使用
rate_limiter = RateLimiter()
