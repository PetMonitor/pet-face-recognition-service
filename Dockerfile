FROM python:3.7

ADD ./requirements.txt /app/requirements.txt
ADD ./src /app/src

WORKDIR /app

ENV PYTHONPATH=/app
ENV LOCAL_PORT=5000

RUN pip3 install -r requirements.txt

EXPOSE $LOCAL_PORT

CMD ["python3", "/app/src/main/service/Service.py"]