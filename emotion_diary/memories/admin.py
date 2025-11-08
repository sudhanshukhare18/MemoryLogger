# memories/admin.py
from django.contrib import admin
from .models import Memory

@admin.register(Memory)
class MemoryAdmin(admin.ModelAdmin):
    list_display = ("user", "short_text", "emotion_label", "sentiment", "created_at")
    search_fields = ("text_content", "emotion_label", "sentiment", "tags")
    list_filter = ("created_at", "emotion_label", "sentiment")
    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        ("User & Timeline", {
            "fields": ("user", "created_at", "updated_at")
        }),
        ("Diary Content", {
            "fields": ("text_content", "emotion_label", "sentiment", "tags", "media")
        }),
        ("AI Metadata", {
            "fields": ("embedding",)
        }),
    )

    def short_text(self, obj):
        return (obj.text_content[:70] + "...") if len(obj.text_content) > 70 else obj.text_content
    short_text.short_description = "Memory Snippet"
