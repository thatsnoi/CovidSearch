# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 python:3.8
COPY / /

RUN pip install wheel
# RUN pip install https://github.com/diyor28/tf-docker-m1/releases/download/v1.0.0/tensorflow-2.8.0-cp39-cp39-linux_aarch64.whl
RUN pip install -r requirements.txt
RUN python3 main.py