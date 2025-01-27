FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ["predict.py", "self_model.pth", "./"]

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
