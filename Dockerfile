# This dockerfile was put together specifically to get
# tensorflow to work in macos with M1 chip.
FROM --platform=linux/amd64 python:3.7
ADD ./requirements.txt /app/requirements.txt
ADD ./src /app/src

WORKDIR /app

ENV PYTHONPATH=/app
ENV LOCAL_PORT=5000

RUN python -m pip install --upgrade pip

RUN pip install -U https://tf.novaal.de/barcelona/tensorflow-2.5.0-cp37-cp37m-linux_x86_64.whl
RUN pip install -r requirements.txt

EXPOSE $LOCAL_PORT

CMD ["python", "/app/src/main/service/Service.py"]