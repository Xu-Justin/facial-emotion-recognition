FROM pytorch/pytorch

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY apps.py .
COPY resources resources
COPY templates templates

EXPOSE 5000

ENTRYPOINT [ "python3", "apps.py" ]