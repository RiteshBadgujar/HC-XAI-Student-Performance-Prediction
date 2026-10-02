from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Student, AcademicRecord, TeacherDecision, Feedback


class SignupForm(UserCreationForm):
    full_name = forms.CharField(max_length=150)
    email = forms.EmailField()
    department = forms.CharField(max_length=150, required=False)

    class Meta:
        model = User
        fields = ("username", "email", "full_name", "department", "password1", "password2")

    def save(self, commit=True):
        user = super().save(commit=False)
        full_name = self.cleaned_data["full_name"].strip().split(maxsplit=1)
        user.first_name = full_name[0] if full_name else ""
        user.last_name = full_name[1] if len(full_name) > 1 else ""
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user


class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ["student_id", "full_name", "class_year", "education_level", "age", "gender", "email", "phone", "department"]


class AcademicRecordForm(forms.ModelForm):
    class Meta:
        model = AcademicRecord
        exclude = ["student", "created_at", "updated_at"]

    def clean(self):
        cleaned = super().clean()
        pairs = (("previous_total_marks", "previous_gained_marks"), ("internal_total_marks", "internal_gained_marks"), ("total_assignments", "completed_assignments"))
        for total, gained in pairs:
            t, g = cleaned.get(total), cleaned.get(gained)
            if t is not None and g is not None and g > t:
                self.add_error(gained, "Obtained/completed value cannot exceed the total.")
        return cleaned


class TeacherDecisionForm(forms.ModelForm):
    class Meta:
        model = TeacherDecision
        fields = ["decision", "comment"]
        widgets = {"comment": forms.Textarea(attrs={"rows": 4})}


class FeedbackForm(forms.ModelForm):
    class Meta:
        model = Feedback
        fields = ["prediction_usefulness", "explanation_understood", "comment"]
        widgets = {"comment": forms.Textarea(attrs={"rows": 4})}


class ProfileForm(forms.Form):
    full_name = forms.CharField(max_length=150)
    department = forms.CharField(max_length=150, required=False)
    phone = forms.CharField(max_length=30, required=False)
    institute = forms.CharField(max_length=200, required=False)


class ChangePasswordForm(forms.Form):
    current_password = forms.CharField(widget=forms.PasswordInput)
    new_password = forms.CharField(min_length=8, widget=forms.PasswordInput)
    confirm_password = forms.CharField(min_length=8, widget=forms.PasswordInput)

    def clean(self):
        cleaned = super().clean()
        if cleaned.get("new_password") != cleaned.get("confirm_password"):
            self.add_error("confirm_password", "Passwords do not match.")
        return cleaned
