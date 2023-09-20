##Builder

FROM python:3.10.6-buster as builder

WORKDIR /prod

RUN python -m venv /opt/venv
ENV PATH = "/opt/venv/bin:$PATH"
RUN . /opt/venv/bin/activate

COPY requirements_api.txt requirements.txt
RUN pip install -r requirements.txt

COPY setup.py setup.py
RUN . /opt/venv/bin/activate && pip install .

COPY bert_binary.h5 bert_binary.h5
COPY utils/ utils/

COPY api api


#Runner
FROM python:3.10.6-slim-buster as runner

WORKDIR /app

ENV PATH = "/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /prod/bert_binary.h5 /app/bert_binary.h5
COPY --from=builder /prod/utils /app/utils
COPY --from=builder /prod/api /app/api

CMD /opt/venv/bin/uvicorn api.fast:app --host 0.0.0.0 --port $PORT
