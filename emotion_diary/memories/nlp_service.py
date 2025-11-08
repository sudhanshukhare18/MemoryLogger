import re
import torch
import numpy as np
from typing import List, Dict, Any
from django.utils import timezone
from transformers import AutoTokenizer, AutoModel, pipeline
import spacy
from sklearn.metrics.pairwise import cosine_similarity

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
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
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
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()
# ---------------------- NLP HELPERS ----------------------
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

# ---------------------- CORE LOGIC ----------------------
def process_and_store_text(text, user, media=None):
    """
    Analyze memory → embed → tag → emotion → sentiment → save safely
    """
    from .models import Memory
    import numpy as np

    try:
        # 🔹 Step 1: Generate embedding
        emb_array = get_embedding_for_text(text)
        embedding = np.array(emb_array).astype(float).tolist()

        # 🔹 Step 2: Detect emotion
        emotion = detect_emotion(text)

        # 🔹 Step 3: Extract tags
        tags = extract_tags(text)

        # 🔹 Step 4: Map emotion → sentiment
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
            "anticipation": "positive"
        }
        sentiment = emotion_to_sentiment.get(emotion.lower(), "neutral")

        # 🔹 Step 5: Save memory in database
        memory = Memory.objects.create(
            user=user,
            text_content=text,
            emotion_label=emotion,
            sentiment=sentiment,
            embedding=embedding,
            tags=tags,
            media=media,
            created_at=timezone.now(),
        )

        print(f"✅ Saved memory {memory.id} with emotion '{emotion}' and sentiment '{sentiment}'")
        return memory

    except Exception as e:
        print(f"❌ Error saving memory: {e}")
        memory = Memory.objects.create(
            user=user,
            text_content=text,
            emotion_label="neutral",
            sentiment="neutral",
            embedding=None,
            created_at=timezone.now(),
        )
        return memory



# ---------------------- SEMANTIC SEARCH ----------------------
def search_and_summarize(query: str, user, top_k: int = 5, min_similarity: float = 0.10) -> Dict[str, Any]:
    from .models import Memory
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from collections import Counter

    result = {"summary": "", "matches": []}
    if not query or not user:
        return result

    # Step 1: Generate embedding for the query
    q_emb = np.array(get_embedding_for_text(query)).reshape(1, -1)

    # Step 2: Fetch all user memories
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

    # Step 3: Compute cosine similarity
    mem_vectors = np.stack([vec for (_, vec) in emb_list])
    sims = cosine_similarity(mem_vectors, q_emb).flatten()

    scored = [(float(s), emb_list[idx][0]) for idx, s in enumerate(sims)]
    filtered = [(s, m) for s, m in sorted(scored, key=lambda x: x[0], reverse=True) if s >= min_similarity][:top_k]

    result["matches"] = filtered
    if not filtered:
        result["summary"] = "No related memories found."
        return result

    # Step 4: Combine matched memory texts
    matched_memories = [m for _, m in filtered]
    combined_text = "\n".join([f"{m.text_content} ({m.emotion_label})" for m in matched_memories])

    # Limit size for summarization
    words = combined_text.split()
    if len(words) > 700:
        combined_text = " ".join(words[:700])

    # Step 5: Detect dominant emotion
    emotion_labels = [m.emotion_label for m in matched_memories if m.emotion_label]
    dominant_emotion = "neutral"
    if emotion_labels:
        dominant_emotion = Counter(emotion_labels).most_common(1)[0][0]

    # Step 6: Map emotion to tone
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

    # Step 7: Clean summarizer-friendly prompt (no meta text)
    prompt = (
        f"These diary excerpts reflect a {tone} mood:\n\n{combined_text}\n\n"
        "Summarize them briefly into one cohesive emotional paragraph."
    )

    # Step 8: Summarize
    try:
        summarizer = get_summarizer()
        out = summarizer(
            prompt,
            max_length=200,
            min_length=60,
            truncation=True,
            do_sample=False
        )
        raw_summary = out[0]["summary_text"].strip()

        # Post-process: capitalize, remove artifacts
        cleaned_summary = raw_summary.replace("diary", "").replace("excerpt", "").strip()
        cleaned_summary = cleaned_summary[0].upper() + cleaned_summary[1:]

        # Add a small emotional closer
        result["summary"] = f"{cleaned_summary} It captures the {tone} spirit of these moments."

    except Exception as e:
        print(f"⚠️ Summarization failed: {e}")
        result["summary"] = combined_text[:400] + "..."

    print(f"✅ Found {len(filtered)} related memories")
    print(f"🧠 Dominant Emotion: {dominant_emotion} → Tone: {tone}")
    return result


def summarize_multiple_memories(memories):
    """
    Given a list of Memory objects, generate a medium-length cohesive summary paragraph.
    """
    if not memories:
        return "No related memories found."

    combined_text = "\n".join([f"{m.text_content} ({m.emotion_label})" for m in memories])
    words = combined_text.split()
    if len(words) > 700:
        combined_text = " ".join(words[:700])  # truncate if too large

    prompt = (
        "You are an empathetic storyteller. Summarize the following diary entries "
        "and emotions into one warm, cohesive medium-length paragraph (6–8 sentences). "
        "Focus on feelings, tone, and emotional flow rather than specific details.\n\n"
        f"{combined_text}"
    )

    try:
        summarizer = get_summarizer()
        out = summarizer(prompt, max_length=200, min_length=80, do_sample=False)
        return out[0]["summary_text"].strip()
    except Exception as e:
        print(f"⚠️ Summarization fallback due to: {e}")
        return combined_text[:400] + "..."
