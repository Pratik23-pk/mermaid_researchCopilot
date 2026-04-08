import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import requests
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST


def _fastapi_headers(request: HttpRequest) -> dict[str, str]:
    return {
        "X-User-Id": str(request.user.id),
    }


def _fastapi_error_response(exc: Exception) -> JsonResponse:
    return JsonResponse(
        {"ok": False, "error": f"Backend request failed: {exc}"},
        status=502,
    )


def _safe_upload_name(name: str) -> str:
    base = os.path.basename(name or "uploaded.pdf")
    chars = []
    for ch in base:
        if ch.isalnum() or ch in {".", "_", "-"}:
            chars.append(ch)
        else:
            chars.append("_")
    safe = "".join(chars)
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    return safe


def _stage_uploaded_pdfs(files) -> tuple[Path, list[str]]:
    staging_root = Path(settings.UPLOAD_STAGING_ROOT)
    staging_dir = staging_root / f"batch_{uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_paths: list[str] = []
    total_bytes = 0
    used_names: set[str] = set()

    try:
        for uploaded in files:
            if not (uploaded.name or "").lower().endswith(".pdf"):
                raise ValueError(f"Only PDF uploads are supported: {uploaded.name}")

            safe_name = _safe_upload_name(uploaded.name)
            base, ext = os.path.splitext(safe_name)
            candidate_name = safe_name
            counter = 1
            while candidate_name in used_names:
                candidate_name = f"{base}_{counter}{ext}"
                counter += 1
            used_names.add(candidate_name)

            destination = staging_dir / candidate_name
            written = 0

            with destination.open("wb") as out:
                for chunk in uploaded.chunks():
                    written += len(chunk)
                    total_bytes += len(chunk)

                    if written > settings.MAX_UPLOAD_FILE_BYTES:
                        raise ValueError(
                            f"{uploaded.name} exceeds the per-file limit of "
                            f"{settings.MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MB."
                        )
                    if total_bytes > settings.MAX_UPLOAD_TOTAL_BYTES:
                        raise ValueError(
                            f"Upload exceeds the total limit of "
                            f"{settings.MAX_UPLOAD_TOTAL_BYTES // (1024 * 1024)} MB."
                        )

                    out.write(chunk)

            staged_paths.append(str(destination))
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return staging_dir, staged_paths


def landing_page(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("workspace")
    return render(request, "landing.html")


@require_POST
def signup_view(request: HttpRequest) -> HttpResponse:
    username = (request.POST.get("username") or "").strip()
    email = (request.POST.get("email") or "").strip()
    password = request.POST.get("password") or ""

    if len(username) < 3:
        messages.error(request, "Username must be at least 3 characters.")
        return redirect("landing")

    if len(password) < 6:
        messages.error(request, "Password must be at least 6 characters.")
        return redirect("landing")

    if User.objects.filter(username=username).exists():
        messages.error(request, "Username already exists.")
        return redirect("landing")

    user = User.objects.create_user(username=username, email=email, password=password)
    login(request, user)
    return redirect("workspace")


@require_POST
def login_view(request: HttpRequest) -> HttpResponse:
    username = (request.POST.get("username") or "").strip()
    password = request.POST.get("password") or ""

    user = authenticate(request, username=username, password=password)
    if user is None:
        messages.error(request, "Invalid credentials.")
        return redirect("landing")

    login(request, user)
    return redirect("workspace")


@login_required
@require_POST
def logout_view(request: HttpRequest) -> HttpResponse:
    try:
        requests.post(
            f"{settings.FASTAPI_BASE_URL}/session/cleanup",
            headers=_fastapi_headers(request),
            timeout=30,
        )
    except Exception:
        # Logout should continue even if cleanup endpoint is temporarily unavailable.
        pass

    logout(request)
    return redirect("landing")


@login_required
@require_GET
def workspace_page(request: HttpRequest) -> HttpResponse:
    return render(request, "workspace.html")


@login_required
@require_POST
def upload_pdfs_view(request: HttpRequest) -> JsonResponse:
    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse({"ok": False, "error": "No files uploaded."}, status=400)

    try:
        staging_dir, staged_paths = _stage_uploaded_pdfs(files)
    except ValueError as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return _fastapi_error_response(exc)

    try:
        try:
            resp = requests.post(
                f"{settings.FASTAPI_BASE_URL}/ingest/staged-pdfs",
                headers={**_fastapi_headers(request), "Content-Type": "application/json"},
                json={"paths": staged_paths},
                timeout=settings.FASTAPI_UPLOAD_TIMEOUT_SECONDS,
            )
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
    except Exception as exc:
        return _fastapi_error_response(exc)

    try:
        data = resp.json()
    except Exception:
        data = {"detail": resp.text}

    if resp.status_code != 200:
        return JsonResponse({"ok": False, "error": data}, status=resp.status_code)

    return JsonResponse({"ok": True, "data": data})


@login_required
@require_POST
def ingest_url_view(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON body."}, status=400)

    url = (payload.get("url") or "").strip()
    if not url:
        return JsonResponse({"ok": False, "error": "URL is required."}, status=400)

    try:
        resp = requests.post(
            f"{settings.FASTAPI_BASE_URL}/ingest/url",
            headers={**_fastapi_headers(request), "Content-Type": "application/json"},
            json={"url": url},
            timeout=60,
        )
    except Exception as exc:
        return _fastapi_error_response(exc)

    try:
        data = resp.json()
    except Exception:
        data = {"detail": resp.text}

    if resp.status_code != 200:
        return JsonResponse({"ok": False, "error": data}, status=resp.status_code)

    return JsonResponse({"ok": True, "data": data})


@login_required
@require_POST
def chat_view(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON body."}, status=400)

    question = (payload.get("question") or "").strip()
    if len(question) < 2:
        return JsonResponse({"ok": False, "error": "Question is too short."}, status=400)

    try:
        resp = requests.post(
            f"{settings.FASTAPI_BASE_URL}/chat",
            headers={**_fastapi_headers(request), "Content-Type": "application/json"},
            json={"question": question},
            timeout=120,
        )
    except Exception as exc:
        return _fastapi_error_response(exc)

    try:
        data = resp.json()
    except Exception:
        data = {"detail": resp.text}

    if resp.status_code != 200:
        return JsonResponse({"ok": False, "error": data}, status=resp.status_code)

    return JsonResponse({"ok": True, "data": data})


@login_required
@require_GET
def stats_view(request: HttpRequest) -> JsonResponse:
    try:
        resp = requests.get(
            f"{settings.FASTAPI_BASE_URL}/session/stats",
            headers=_fastapi_headers(request),
            timeout=30,
        )
    except Exception as exc:
        return _fastapi_error_response(exc)

    try:
        data = resp.json()
    except Exception:
        data = {"detail": resp.text}

    if resp.status_code != 200:
        return JsonResponse({"ok": False, "error": data}, status=resp.status_code)

    return JsonResponse({"ok": True, "data": data})
