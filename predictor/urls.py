from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

urlpatterns = [
    path("", views.landing, name="landing"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("signup/", views.signup, name="signup"),
    path("students/", views.student_list, name="student_list"),
    path("report/<str:student_id>/", views.report_detail, name="report_detail"),
    path("predict/", views.predict_performance, name="predict_performance"),
    path("compare/", views.compare_students, name="compare_students"),
    path("analytics/", views.class_analytics, name="class_analytics"),
    path("login/", auth_views.LoginView.as_view(template_name="predictor/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("profile/", views.profile, name="profile"),
]