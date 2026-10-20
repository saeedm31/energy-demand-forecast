FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Train once at build time so the container starts instantly
RUN python -c "from energy_forecast import service, data; y, i = data.load(); m = service.load_or_train(y, i); import joblib, os; os.makedirs('models', exist_ok=True); joblib.dump(m, 'models/model.joblib')"

EXPOSE 8000
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8000", "api:app"]
