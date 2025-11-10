from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from .models import Memory
from .form import MemoryForm
from . import nlp_service

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

# --------------------------------------------------
# 📝 Add New Memory (with NLP embedding + tags)
# --------------------------------------------------
@login_required
def add_memory(request):
    if request.method == "POST":
        text = request.POST.get("text_content", "").strip()
        print("📥 Got memory text:", text)

        if not text:
            return JsonResponse({"error": "No text received"}, status=400)

        try:
            # Process NLP + store in DB
            memory = nlp_service.process_and_store_text(text=text, user=request.user)
            print(f"✅ Memory saved with ID {memory.id}, tags: {memory.tags}")
            return JsonResponse({"message": "Memory saved successfully!"})
        except Exception as e:
            print("❌ Error while saving memory:", e)
            return JsonResponse({"error": str(e)}, status=500)

    # For GET requests, just show the memory form page
    return render(request, "index.html")


# --------------------------------------------------
# 📖 View All Memories
# --------------------------------------------------
@login_required
def view_memories(request):
    query = request.GET.get('q', '')
    memories = Memory.objects.filter(user=request.user).order_by('-created_at')

    if query:
        memories = memories.filter(text_content__icontains=query)

    return render(request, 'memories.html', {'memories': memories})


# --------------------------------------------------
# 🔍 Semantic Search + Narrative Summary
# --------------------------------------------------
@login_required
def search_summary(request):
    query = request.GET.get("q", "").strip()
    result = {"summary": "", "matches": []}

    if query:
        result = nlp_service.search_and_summarize(query=query, user=request.user, top_k=1)

    return render(request, "search_summary.html", {
        "query": query,
        "summary": result.get("summary", ""),
        "matches": result.get("matches", []),
        "today": timezone.now().strftime("%A, %B %d, %Y"),
    })
# 🧠 Add new memory via API
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_add_memory(request):
    """
    Create a new memory using JWT authentication.
    Example request:
    {
        "text": "Had a great evening walk with Isha by the lake."
    }
    """
    text = request.data.get("text", "").strip()
    if not text:
        return Response({"error": "No text provided."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        memory = nlp_service.process_and_store_text(text=text, user=request.user)
        return Response({
            "id": memory.id,
            "emotion": memory.emotion_label,
            "sentiment": memory.sentiment,
            "tags": memory.tags,
            "created_at": memory.created_at,
            "message": "Memory added successfully."
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        print("❌ API Error while saving memory:", e)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# 🔍 Search memories semantically via API
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_search_memories(request):
    """
    Search user memories semantically and get AI-generated summary.
    Example request:
    /memories/api/search/?q=college+trip
    """
    query = request.GET.get("q", "").strip()
    if not query:
        return Response({"error": "Please provide a query."}, status=status.HTTP_400_BAD_REQUEST)

    result = nlp_service.search_and_summarize(query=query, user=request.user)
    matches = [
        {
            "id": m.id,
            "text": m.text_content,
            "emotion": m.emotion_label,
            "sentiment": m.sentiment,
            "created_at": m.created_at
        } for _, m in result.get("matches", [])
    ]

    return Response({
        "query": query,
        "summary": result.get("summary", ""),
        "matches": matches
    }, status=status.HTTP_200_OK)