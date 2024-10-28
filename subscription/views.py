from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from django.apps import apps
from .models import RedeemCode, Subscription
from .serializers import RedeemCodeSerializer, SubscriptionSerializer

class RedeemCodeViewSet(viewsets.ModelViewSet):
    queryset = RedeemCode.objects.all()
    serializer_class = RedeemCodeSerializer

    @action(detail=False, methods=['post'])
    def redeem(self, request):
        code = request.data.get('code')
        try:
            redeem_code = RedeemCode.objects.get(code=code, is_used=False)
        except RedeemCode.DoesNotExist:
            return Response({'error': 'Invalid or used code'}, status=status.HTTP_400_BAD_REQUEST)

        user = request.user
        user.profile.balance += redeem_code.amount
        user.profile.save()

        redeem_code.is_used = True
        redeem_code.used_by = user
        redeem_code.used_at = timezone.now()
        redeem_code.save()

        return Response({'success': True, 'new_balance': user.profile.balance})

class SubscriptionViewSet(viewsets.ModelViewSet):
    queryset = Subscription.objects.all()
    serializer_class = SubscriptionSerializer

    @action(detail=True, methods=['post'])
    def purchase(self, request, pk=None):
        subscription = self.get_object()
        user = request.user

        if user.profile.balance < subscription.price:
            return Response({'error': '余额不足'}, status=status.HTTP_400_BAD_REQUEST)

        user.profile.balance -= subscription.price
        user.profile.subscription = subscription
        user.profile.subscription_expiry = timezone.now() + subscription.duration
        user.profile.subscription_is_active = True
        user.profile.save()

        # 更新内存字典
        app_config = apps.get_app_config('subscription')
        if hasattr(app_config, 'subscription_checker'):
            with app_config.subscription_checker.lock:
                app_config.subscription_checker.subscriptions[user.id] = user.profile.subscription_expiry

        return Response({
            'success': True,
            'new_balance': user.profile.balance,
            'subscription': SubscriptionSerializer(subscription).data
        })

class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        profile = user.profile
        redeem_codes = RedeemCode.objects.filter(used_by=user)
        current_subscription = profile.subscription

        data = {
            'balance': profile.balance,
            'redeem_codes': RedeemCodeSerializer(redeem_codes, many=True).data,
            'current_subscription': SubscriptionSerializer(current_subscription).data if current_subscription else None,
            'subscription_expiry': profile.subscription_expiry
        }

        return Response(data)

