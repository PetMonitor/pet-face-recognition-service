FROM python:3.7

ADD ./requirements.txt /app/requirements.txt
ADD ./src /app/src

WORKDIR /app

ENV PYTHONPATH=/app

RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD ["python3", "/app/src/main/service/Service.py"]