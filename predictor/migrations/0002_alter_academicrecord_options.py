"""
Migration 0002 — User-wise data isolation for Student model.

Safe steps:
1. Drop the old global unique index on student_id.
2. Add owner FK (nullable=True so existing rows are not rejected).
3. Assign existing students to the first superuser (preserves all data).
4. Add the new per-owner unique constraint (owner, student_id).

Existing data is fully preserved. No rows are deleted.
"""
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


def assign_existing_students_to_first_superuser(apps, schema_editor):
    """
    Assign all existing students without an owner to the first superuser.
    If no superuser exists, assign to the first user. If no users exist at all,
    leave owner=NULL (safe because the column is nullable).
    """
    User = apps.get_model(settings.AUTH_USER_MODEL)
    Student = apps.get_model("predictor", "Student")

    owner = (
        User.objects.filter(is_superuser=True).order_by("id").first()
        or User.objects.order_by("id").first()
    )
    if owner is None:
        return  # No users yet — leave owner NULL

    Student.objects.filter(owner__isnull=True).update(owner=owner)


class Migration(migrations.Migration):

    dependencies = [
        ("predictor", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        # 1. Remove the old globally-unique constraint on student_id
        migrations.AlterField(
            model_name="student",
            name="student_id",
            field=models.CharField(max_length=50),  # unique=True removed
        ),

        # 2. Add the owner FK (nullable so existing rows survive)
        migrations.AddField(
            model_name="student",
            name="owner",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="students",
                to=settings.AUTH_USER_MODEL,
            ),
        ),

        # 3. Assign existing students to the first superuser
        migrations.RunPython(
            assign_existing_students_to_first_superuser,
            reverse_code=migrations.RunPython.noop,
        ),

        # 4. Add per-owner unique constraint: same student_id is fine for different teachers
        migrations.AddConstraint(
            model_name="student",
            constraint=models.UniqueConstraint(
                fields=["owner", "student_id"],
                name="predictor_student_owner_student_id_uniq",
            ),
        ),
    ]
