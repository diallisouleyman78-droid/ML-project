FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN apt update && apt install -y awscli && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
CMD ["python", "app.py"]