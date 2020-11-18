FROM python:3.6

COPY ./Files/requirements.txt /

RUN pip install -r /requirements.txt

WORKDIR /Files

ADD . /Files

CMD ["python", "./Files/SampleTest.py"]
