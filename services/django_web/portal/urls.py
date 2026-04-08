from django.urls import path

from . import views

urlpatterns = [
    path("", views.landing_page, name="landing"),
    path("auth/signup/", views.signup_view, name="signup"),
    path("auth/login/", views.login_view, name="login"),
    path("auth/logout/", views.logout_view, name="logout"),
    path("workspace/", views.workspace_page, name="workspace"),
    path("api/upload-pdfs/", views.upload_pdfs_view, name="upload_pdfs"),
    path("api/ingest-url/", views.ingest_url_view, name="ingest_url"),
    path("api/chat/", views.chat_view, name="chat"),
    path("api/stats/", views.stats_view, name="stats"),
]
