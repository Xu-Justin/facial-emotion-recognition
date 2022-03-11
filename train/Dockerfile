FROM pytorch/pytorch

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY train.py .

ENTRYPOINT [ "python3", "train.py" ]