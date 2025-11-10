import re
import torch
import numpy as np
from typing import List, Dict, Any
from django.utils import timezone
from transformers import AutoTokenizer, AutoModel, pipeline
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from .encryption import encrypt_aes, decrypt_aes

# Lazy global loads
_nlp = None
_summarizer = None
_emotion_pipe = None

# ---------------------- MODEL LOADERS ----------------------
def get_spacy():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer


def get_emotion_pipe():
    global _emotion_pipe
    if _emotion_pipe is None:
        _emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
    return _emotion_pipe


# ---------------------- EMBEDDING MODEL ----------------------
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def get_embedding(text: str) -> list:
    """Return embedding vector for text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1)
    return emb[0].numpy().tolist()


def get_embedding_for_text(text: str):
    """Generate embedding vector for a given text"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()


# ---------------------- HELPERS ----------------------
def extract_tags(text):
    """Extract simple keywords for tags."""
    words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
    common = {"the","and","but","are","you","for","this","that","was","have","with"}
    tags = [w for w in words if w not in common]
    return list(set(tags))[:10]


def detect_emotion(text: str) -> str:
    try:
        pipe = get_emotion_pipe()
        res = pipe(text[:512])
        if res and isinstance(res, list):
            return res[0]["label"]
    except Exception:
        return "neutral"
    return "neutral"


def is_probably_encrypted(value: str) -> bool:
    """Heuristic: check if a string looks like base64 AES ciphertext."""
    if not isinstance(value, str):
        return False
    if len(value) < 24:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", value):
        return False
    return True


# ---------------------- CORE: PROCESS AND STORE ----------------------
def process_and_store_text(text, user, media=None):
    """
    Analyze memory → embed → tag → emotion → sentiment → encrypt → store
    """
    from .models import Memory
    import numpy as np

    try:
        # 🔹 Step 1: Embedding
        emb_array = get_embedding_for_text(text)
        embedding = np.array(emb_array).astype(float).tolist()

        # 🔹 Step 2: Emotion detection
        emotion = detect_emotion(text)

        # 🔹 Step 3: Tags
        tags = extract_tags(text)

        # 🔹 Step 4: Sentiment mapping
        emotion_to_sentiment = {
            "joy": "positive",
            "happiness": "positive",
            "love": "positive",
            "surprise": "neutral",
            "neutral": "neutral",
            "sadness": "negative",
            "anger": "negative",
            "fear": "negative",
            "disgust": "negative",
            "trust": "positive",
            "anticipation": "positive",
        }
        sentiment = emotion_to_sentiment.get(emotion.lower(), "neutral")

        # 🔹 Step 5: Encrypt sensitive fields
        enc_text = encrypt_aes(text)
        enc_emotion = encrypt_aes(emotion)
        enc_sentiment = encrypt_aes(sentiment)

        # 🔹 Step 6: Save to DB
        memory = Memory.objects.create(
            user=user,
            text_content=enc_text,
            emotion_label=enc_emotion,
            sentiment=enc_sentiment,
            embedding=embedding,
            tags=tags,
            media=media,
            created_at=timezone.now(),
        )

        print(f"✅ Saved memory {memory.id} with emotion '{emotion}' and sentiment '{sentiment}'")
        return memory

    except Exception as e:
        print(f"❌ Error saving memory: {e}")
        return Memory.objects.create(
            user=user,
            text_content=text,
            emotion_label="neutral",
            sentiment="neutral",
            embedding=None,
            created_at=timezone.now(),
        )


# ---------------------- SEMANTIC SEARCH ----------------------
def search_and_summarize(query: str, user, top_k: int = 1, min_similarity: float = 0.10) -> Dict[str, Any]:
    from .models import Memory
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from collections import Counter

    result = {"summary": "", "matches": []}
    if not query or not user:
        return result

    q_emb = np.array(get_embedding_for_text(query)).reshape(1, -1)
    memories = Memory.objects.filter(user=user)

    emb_list = []
    for m in memories:
        if m.embedding:
            try:
                emb = np.array(m.embedding, dtype=float)
                emb_list.append((m, emb))
            except Exception:
                continue

    if not emb_list:
        result["summary"] = "No embedded memories found."
        return result

    mem_vectors = np.stack([v for (_, v) in emb_list])
    sims = cosine_similarity(mem_vectors, q_emb).flatten()
    scored = [(float(s), emb_list[i][0]) for i, s in enumerate(sims)]
    filtered = [(s, m) for s, m in sorted(scored, key=lambda x: x[0], reverse=True) if s >= min_similarity][:top_k]

    if not filtered:
        result["summary"] = "No related memories found."
        return result

    # 🔹 Step 4: Decrypt or fallback
    matched_memories = []
    for _, m in filtered:
        try:
            if looks_like_aes_ciphertext(m.text_content):
                print(f"🔒 AES-encrypted text detected for memory {m.id}")
                text_val = decrypt_aes(m.text_content)
            else:
                print(f"📝 Plaintext detected for memory {m.id}")
                text_val = m.text_content

        except Exception as e:
            print(f"❌ Text decryption failed for memory {m.id}: {e}")
            text_val = m.text_content or "[Decryption Skipped]"


        try:
            if is_probably_encrypted(m.emotion_label):
                emotion_val = decrypt_aes(m.emotion_label)
            else:
                emotion_val = m.emotion_label
        except Exception:
            emotion_val = "neutral"

        matched_memories.append((m, text_val, emotion_val))

    # Combine for summarization
    combined_text = "\n".join([f"{text} ({emotion})" for _, text, emotion in matched_memories])

    # 🔹 Step 5: Detect dominant emotion
    emotion_labels = [e for _, _, e in matched_memories if e]
    dominant_emotion = Counter(emotion_labels).most_common(1)[0][0] if emotion_labels else "neutral"

    # 🔹 Step 6: Emotion → tone map
    tone_map = {
        "joy": "heartwarming and cheerful",
        "happiness": "light and uplifting",
        "sadness": "gentle and nostalgic",
        "anger": "intense and emotional",
        "fear": "thoughtful and cautious",
        "love": "romantic and affectionate",
        "surprise": "curious and amazed",
        "neutral": "balanced and calm",
    }
    tone = tone_map.get(dominant_emotion.lower(), "warm and human")

    # 🔹 Step 7: Summarize
    prompt = f"These diary excerpts reflect a {tone} mood:\n\n{combined_text}\n\nSummarize them briefly into one cohesive emotional paragraph. Output will be like you summarize a person life."

    try:
        summarizer = get_summarizer()
        out = summarizer(prompt, max_length=200, min_length=60, truncation=True, do_sample=False)
        raw_summary = out[0]["summary_text"].strip()
        cleaned_summary = raw_summary.replace("diary", "").replace("excerpt", "").strip()
        cleaned_summary = cleaned_summary[0].upper() + cleaned_summary[1:]
        result["summary"] = f"{cleaned_summary} It captures the {tone} spirit of these moments."
    except Exception as e:
        print(f"⚠️ Summarization failed: {e}")
        result["summary"] = combined_text[:400] + "..."

    print(f"✅ Found {len(filtered)} related memories")
    print(f"🧠 Dominant Emotion: {dominant_emotion} → Tone: {tone}")
    # Keep the old structure: return list of (memory, decrypted_text, decrypted_emotion)    
    result["matches"] = [(m, text, emotion) for (m, text, emotion) in matched_memories]

    return result
import base64

def looks_like_aes_ciphertext(value: str) -> bool:
    """Check if a string seems to be AES-encrypted base64."""
    if not value or not isinstance(value, str):
        return False
    if len(value) < 16:
        return False
    try:
        decoded = base64.b64decode(value, validate=True)
        # AES ciphertexts usually at least 16 bytes (IV + tag + data)
        return len(decoded) > 16
    except Exception:
        return False
