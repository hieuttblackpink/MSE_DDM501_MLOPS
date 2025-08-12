FROM python:3.13.5-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
EXPOSE 5050
CMD ["flask", "--app", "app/app.py", "run", "--host=0.0.0.0", "--port=5050"]