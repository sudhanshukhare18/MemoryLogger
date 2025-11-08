# memories/models.py
from django.conf import settings
from django.db import models
from django.utils import timezone
import numpy as np
import json

try:
    from django.db.models import JSONField
except ImportError:
    from django.contrib.postgres.fields import JSONField


class Memory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='memories')
    text_content = models.TextField()
    emotion_label = models.CharField(max_length=64, null=True, blank=True)
    embedding = JSONField(null=True, blank=True)  # ✅ stores vector embedding (list of floats)
    tags = models.JSONField(default=list, blank=True)  # ✅ NLP-generated tags like ["school","friends","happy"]
    sentiment = models.CharField(max_length=20, blank=True)
    media = models.FileField(upload_to='memories_media/', null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    # Helper functions for embedding manipulation
    def set_embedding(self, vector):
        """Save numpy vector as list in JSON."""
        if isinstance(vector, np.ndarray):
            self.embedding = vector.tolist()
        elif isinstance(vector, list):
            self.embedding = vector

    def get_embedding(self):
        """Return embedding as numpy array."""
        if self.embedding:
            return np.array(self.embedding)
        return None

    def __str__(self):
        return f"{self.user.username} — {self.text_content[:50]}"
