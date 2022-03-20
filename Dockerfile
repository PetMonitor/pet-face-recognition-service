# This dockerfile was put together specifically to get
# tensorflow to work in macos with M1 chip.
FROM --platform=linux/amd64 python:3.7
ADD ./requirements.txt /app/requirements.txt
#ADD ./src /app/src

WORKDIR /app

ENV PYTHONPATH=/app
ENV LOCAL_PORT=5001

RUN python3 -m pip install --upgrade pip

#RUN python3 -m install Keras
#RUN pip install -U https://tf.novaal.de/barcelona/tensorflow-2.5.0-cp37-cp37m-linux_x86_64.whl
#RUN pip install -U https://tf.novaal.de/barcelona/tensorflow-2.7.0-cp37-cp37m-linux_x86_64.whl
RUN python3 -m pip install -U https://tf.novaal.de/barcelona/tensorflow-2.8.0-cp37-cp37m-linux_x86_64.whl
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install keras==2.8.0
RUN python3 -m pip install numpy
#RUN python3 -m pip install scikit-learn

EXPOSE $LOCAL_PORT

CMD ["python", "/app/src/main/service/Service.py"]