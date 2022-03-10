FROM pytorch/pytorch

WORKDIR /app
COPY train.py .
COPY requirements.txt .

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3", "app-flask.py" ]