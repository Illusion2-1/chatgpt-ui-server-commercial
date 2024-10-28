from django.contrib import admin
from django.urls import path
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Subscription, Profile, RedeemCode
import uuid
import csv
from django.forms import JSONField as JSONFormField

@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ('title', 'price', 'duration', 'rate_limit', 'device_limit')
    search_fields = ('title',)
    
    formfield_overrides = {
        JSONFormField: {'widget': JSONFormField},
    }

    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        form.base_fields['available_models'].help_text = '请输入有效的JSON格式，例如：["gpt-3.5-turbo", "gpt-4"]'
        return form

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'balance', 'subscription', 'subscription_expiry', 'subscription_is_active')
    search_fields = ('user__username',)
    list_filter = ('subscription_is_active',)

@admin.register(RedeemCode)
class RedeemCodeAdmin(admin.ModelAdmin):
    list_display = ('code', 'amount', 'is_used', 'used_by', 'used_at')
    search_fields = ('code',)
    list_filter = ('is_used',)
    
    # 指定自定义的 change list 模板
    change_list_template = "admin/subscription/redeem_code_changelist.html"
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('generate_codes/', self.generate_codes_view, name='generate_codes'),
        ]
        return custom_urls + urls
    
    def generate_codes_view(self, request):
        if request.method == 'POST':
            amount = float(request.POST.get('amount'))
            count = int(request.POST.get('count'))
            
            for _ in range(count):
                RedeemCode.objects.create(
                    code=str(uuid.uuid4()),
                    amount=amount
                )
            
            self.message_user(request, f"成功生成 {count} 个兑换码")
            return redirect('..')
        
        return render(request, 'admin/subscription/generate_codes.html')
    
    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context['show_generate_codes_button'] = True
        return super().changelist_view(request, extra_context=extra_context)
    
    actions = ['export_selected_codes']

    def export_selected_codes(self, request, queryset):
        response = HttpResponse(content_type='text/plain')
        response['Content-Disposition'] = 'attachment; filename="redeem_codes.txt"'
        
        writer = csv.writer(response)
        for code in queryset:
            writer.writerow([f"{code.code},{code.amount}"])
        
        return response
    
    export_selected_codes.short_description = "导出选中的兑换码"

