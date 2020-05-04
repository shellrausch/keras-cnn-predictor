FROM python:3-slim

ADD . /prediction
WORKDIR /prediction

RUN pip install -r requirements.txt

CMD [ "python" , "-u", "predictor.py"]
