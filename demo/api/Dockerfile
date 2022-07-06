# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 python:3.9

WORKDIR /code

COPY / /code

RUN pip install wheel

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
