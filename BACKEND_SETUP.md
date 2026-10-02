# HC-XAI Backend Foundation

This version replaces the previous hardcoded backend with database-backed Django models and services.

## 1. Install dependencies

```cmd
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configure PostgreSQL

Create a PostgreSQL database named `hc_xai_db` or set `DB_NAME` in your environment.

The default development values are:

- host: 127.0.0.1
- port: 5432
- database: hc_xai_db
- user: postgres
- password: root

Prefer environment variables rather than storing production credentials in source code.

## 3. Run migrations

```cmd
python manage.py migrate
```

## 4. Create an admin account

```cmd
python manage.py createsuperuser
```

## 5. Start the server

```cmd
python manage.py runserver
```

## 6. ML model training

The application does not invent a prediction model. A labeled historical dataset is required.

The training CSV must contain the following feature columns:

`previous_percent, internal_percent, assignment_percent, attendance, study_hours, participation, backlogs, past_failures, sleep_hours, tutoring_sessions`

and a target column named `performance_category` by default.

Example:

```cmd
python manage.py train_model path\to\training_data.csv
```

You can use another target column:

```cmd
python manage.py train_model path\to\training_data.csv --target performance_category
```

The model is saved to:

`predictor/artifacts/student_performance_model.joblib`

## 7. Prediction behavior

Prediction is blocked until the trained model exists. Missing academic fields are reported instead of silently fabricating values.

SHAP is calculated from the actual tree model and the actual student input when SHAP is installed.

The teacher remains the final decision-maker.
