from django.contrib.auth import views as auth_views
from django.urls import path
from . import views

urlpatterns = [
    path("", views.landing, name="landing"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("signup/", views.signup, name="signup"),
    path("login/", views.login_view, name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("profile/", views.profile, name="profile"),
    path("profile/password/", views.change_password, name="change_password"),

    path("students/", views.student_list, name="student_list"),
    path("students/add/", views.student_create, name="student_create"),
    path("students/<str:student_id>/", views.student_detail, name="student_detail"),
    path("students/<str:student_id>/edit/", views.student_update, name="student_update"),
    path(
        "students/<str:student_id>/semester/<int:semester>/academic/",
        views.academic_record_edit,
        name="academic_record_edit"
    ),

    path("predict/", views.predict_performance, name="predict_performance"),

    path("report/<str:student_id>/", views.report_detail, name="report_detail"),
    path("report/<str:student_id>/pdf/", views.report_pdf, name="report_pdf"),

    path("compare/", views.compare_students, name="compare_students"),
    path("api/compare/", views.compare_students_api, name="compare_students_api"),
    path("analytics/", views.class_analytics, name="class_analytics"),

    path(
        "prediction/<int:prediction_id>/decision/",
        views.teacher_decision,
        name="teacher_decision"
    ),
    path(
        "prediction/<int:prediction_id>/feedback/",
        views.feedback,
        name="feedback"
    ),
]