from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import RedeemCodeViewSet, SubscriptionViewSet, UserProfileView

router = DefaultRouter()
router.register(r'redeem-codes', RedeemCodeViewSet)
router.register(r'subscriptions', SubscriptionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('user-profile/', UserProfileView.as_view(), name='user-profile'),

]
