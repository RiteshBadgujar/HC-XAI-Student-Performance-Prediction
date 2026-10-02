from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class TeacherProfile(models.Model):
    ROLE_TEACHER = "teacher"
    ROLE_ADMIN = "admin"
    ROLE_CHOICES = ((ROLE_TEACHER, "Teacher"), (ROLE_ADMIN, "Admin"))

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="teacher_profile")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_TEACHER)
    department = models.CharField(max_length=150, blank=True)
    phone = models.CharField(max_length=30, blank=True)
    institute = models.CharField(max_length=200, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username} ({self.role})"


class Student(models.Model):
    student_id = models.CharField(max_length=50, unique=True)
    full_name = models.CharField(max_length=150)
    class_year = models.CharField(max_length=100, blank=True)
    education_level = models.CharField(max_length=50, blank=True)
    age = models.PositiveSmallIntegerField(null=True, blank=True, validators=[MinValueValidator(5), MaxValueValidator(60)])
    gender = models.CharField(max_length=30, blank=True)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=30, blank=True)
    department = models.CharField(max_length=150, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["full_name", "student_id"]

    def __str__(self):
        return f"{self.student_id} - {self.full_name}"


class AcademicRecord(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="academic_records")
    semester = models.PositiveSmallIntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(20)])

    previous_total_marks = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0)])
    previous_gained_marks = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0)])
    internal_total_marks = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0)])
    internal_gained_marks = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0)])
    total_assignments = models.PositiveIntegerField(null=True, blank=True)
    completed_assignments = models.PositiveIntegerField(null=True, blank=True)

    backlogs = models.PositiveIntegerField(default=0)
    past_failures = models.PositiveIntegerField(default=0)
    attendance = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(100)])
    study_hours = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(168)])
    sleep_hours = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(24)])
    learning_mode = models.CharField(max_length=30, blank=True)
    tutoring_sessions = models.PositiveIntegerField(default=0)
    internet_access = models.CharField(max_length=10, blank=True)
    participation = models.FloatField(null=True, blank=True, validators=[MinValueValidator(1), MaxValueValidator(10)])
    motivation_level = models.CharField(max_length=20, blank=True)
    parental_involvement = models.CharField(max_length=20, blank=True)
    extracurricular = models.CharField(max_length=10, blank=True)
    teacher_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["student_id", "semester"]
        constraints = [
            models.UniqueConstraint(fields=["student", "semester"], name="predictor_academicrecord_student_semester_uniq")
        ]

    @property
    def previous_percent(self):
        if self.previous_total_marks and self.previous_gained_marks is not None:
            return round((self.previous_gained_marks / self.previous_total_marks) * 100, 2)
        return None

    @property
    def internal_percent(self):
        if self.internal_total_marks and self.internal_gained_marks is not None:
            return round((self.internal_gained_marks / self.internal_total_marks) * 100, 2)
        return None

    @property
    def assignment_percent(self):
        if self.total_assignments and self.completed_assignments is not None:
            return round((self.completed_assignments / self.total_assignments) * 100, 2)
        return None

    def __str__(self):
        return f"{self.student.student_id} - Semester {self.semester}"


class Prediction(models.Model):
    CATEGORY_CHOICES = (
        ("Excellent", "Excellent"),
        ("Good", "Good"),
        ("Average", "Average"),
        ("Needs Improvement", "Needs Improvement"),
        ("At Risk", "At Risk"),
    )
    RISK_CHOICES = (("Low", "Low"), ("Medium", "Medium"), ("High", "High"))

    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="predictions")
    academic_record = models.ForeignKey(AcademicRecord, on_delete=models.CASCADE, related_name="predictions")
    predicted_category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    predicted_percentage = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(100)])
    confidence = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(100)])
    risk_level = models.CharField(max_length=20, choices=RISK_CHOICES)
    probabilities = models.JSONField(default=dict, blank=True)
    model_name = models.CharField(max_length=150, default="RandomForestClassifier")
    explanation_summary = models.TextField(blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="created_predictions")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.student.student_id} - {self.predicted_category}"


class SHAPExplanation(models.Model):
    prediction = models.OneToOneField(Prediction, on_delete=models.CASCADE, related_name="shap_explanation")
    values = models.JSONField(default=list, blank=True)
    global_importance = models.JSONField(default=list, blank=True)
    local_features = models.JSONField(default=list, blank=True)
    summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class TeacherDecision(models.Model):
    ACCEPT = "accepted"
    REVIEW = "needs_review"
    DECISIONS = ((ACCEPT, "Accept Prediction"), (REVIEW, "Needs Review"))

    prediction = models.OneToOneField(Prediction, on_delete=models.CASCADE, related_name="teacher_decision")
    teacher = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="teacher_decisions")
    decision = models.CharField(max_length=30, choices=DECISIONS)
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class Feedback(models.Model):
    USEFULNESS_CHOICES = (
        ("very_useful", "Very Useful"),
        ("useful", "Useful"),
        ("neutral", "Neutral"),
        ("not_useful", "Not Useful"),
        ("not_useful_at_all", "Not Useful At All"),
    )

    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, related_name="feedback_entries")
    teacher = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="feedback_entries")
    prediction_usefulness = models.CharField(max_length=30, choices=USEFULNESS_CHOICES)
    explanation_understood = models.BooleanField()
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Report(models.Model):
    REPORT_TYPES = (
        ("student_performance", "Student Performance Report"),
        ("prediction", "Prediction Report"),
        ("at_risk", "At-Risk Student Report"),
        ("shap", "SHAP Explanation Report"),
        ("feedback", "Teacher Feedback Report"),
    )

    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="reports")
    prediction = models.ForeignKey(Prediction, null=True, blank=True, on_delete=models.SET_NULL, related_name="reports")
    report_type = models.CharField(max_length=40, choices=REPORT_TYPES, default="student_performance")
    file = models.FileField(upload_to="reports/%Y/%m/", blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="reports")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
