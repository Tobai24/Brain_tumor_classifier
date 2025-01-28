FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY [ "app.py",  "self_model.pth" , "./" ]

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]




