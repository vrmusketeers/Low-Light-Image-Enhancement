FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3

COPY requirements.txt requirements.txt 

COPY app app
COPY app/core app/core
COPY app/core/utils app/core/utils

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /app

ENV FLASK_APP /app/wsgi.py

CMD gunicorn --bind 0.0.0.0:5050 --timeout=600 wsgi:app -w 4

EXPOSE 5050