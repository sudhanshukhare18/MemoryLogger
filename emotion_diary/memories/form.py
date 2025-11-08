from django import forms
from memories.models import Memory

class MemoryForm(forms.ModelForm):
    class Meta:
        model = Memory
        fields = ['text_content', 'emotion_label', 'media']
        widgets = {
            'text_content': forms.Textarea(attrs={
                'rows': 5,
                'placeholder': 'Write about your day or a special moment...',
                'style': 'width:100%; background:#222; color:#fff; border:1px solid #444; border-radius:8px; padding:10px;'
            }),
            'emotion_label': forms.TextInput(attrs={
                'placeholder': 'e.g., Happy, Sad, Excited',
                'style': 'width:100%; background:#222; color:#fff; border:1px solid #444; border-radius:8px; padding:8px;'
            })
        }
def rsa_ed
plaintext = private_key.decrypt(
    cipertext,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)
def enc(a){
    return aes_ed(a)
}
