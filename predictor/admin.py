from django.contrib import admin
from .models import AcademicRecord, Feedback, Prediction, Report, SHAPExplanation, Student, TeacherDecision, TeacherProfile

admin.site.register([Student, AcademicRecord, Prediction, SHAPExplanation, TeacherDecision, Feedback, Report, TeacherProfile])
