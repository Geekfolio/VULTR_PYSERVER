FROM python:3.12-alpine

WORKDIR /app

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD server.py .
ADD gunicorn.conf.py .

CMD [ "gunicorn",  "-c", "gunicorn.conf.py", "server:app" ]