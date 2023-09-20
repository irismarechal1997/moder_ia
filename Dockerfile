##Builder

FROM python:3.10.6-buster as builder

WORKDIR /prod

COPY requirements_api.txt requirements.txt
RUN pip install -r requirements.txt

COPY setup.py setup.py
RUN mkdir pkg
RUN pip install -t pkg .

COPY bert_binary.h5 bert_binary.h5
COPY utils/ utils/

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT

#Runner
FROM python:3.10.6-slim-buster as runner

WORKDIR /app
COPY --from=builder /prod/pkg /app/pkg
COPY --from=builder /prod/bert_binary.h5 /app/bert_binary.h5
COPY --from=builder /prod/utils /app/utils
COPY --from=builder /prod/api.fast.py /app/api.fast.py

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
