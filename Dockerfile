FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY setup.py setup.py
RUN pip install .

COPY bert_binary.h5 bert_binary.h5
COPY utils/ utils/

COPY . .

CMD uvicorn api.fast:app --host 0.0.0.0 --port 5000
# $DEL_END
# #avec GCP $PORT
