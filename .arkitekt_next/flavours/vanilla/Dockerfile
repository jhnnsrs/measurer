FROM python:3.12-slim

RUN pip install "arkitekt-next[all, kraph]>=0.26.0"

RUN mkdir /app
WORKDIR /app
COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py
