FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pybabel compile -d locale -D django || true
RUN python manage.py collectstatic --noinput
EXPOSE 8000
CMD ["gunicorn", "clustmetalearn_web.wsgi:application", "--bind", "0.0.0.0:8000"]