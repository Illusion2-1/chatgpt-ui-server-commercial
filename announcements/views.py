from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from .models import Announcement, AnnouncementSetting
from .serializers import AnnouncementSerializer

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_active_announcements(request):
    current_time = timezone.now()
    settings = AnnouncementSetting.objects.filter(is_active=True, display_until__gte=current_time)
    announcements = Announcement.objects.filter(setting__in=settings).order_by('-created_at')
    serializer = AnnouncementSerializer(announcements, many=True)
    return Response({'announcements': serializer.data})