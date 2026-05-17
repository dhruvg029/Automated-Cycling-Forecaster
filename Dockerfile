FROM python:3.12
COPY requirements.txt /app/requirements.txt
COPY src /app/src

RUN pip install -r /app/requirements.txt