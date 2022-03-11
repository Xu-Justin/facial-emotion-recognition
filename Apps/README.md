# `facial-emotion-recognition:apps`

## Build Image

```
$ docker build -t jstnxu/facial-emotion-recognition:apps .
```

## Pull Image from Docker Hub

```
$ docker pull jstnxu/facial-emotion-recognition:apps
```

## Run Image

To run this image, run the following code. Change `{local-port}` with your local port (or just use 5000).

```
$ docker run --gpus all -p {local-port}:5000 jstnxu/facial-emotion-recognition:apps
```

After running the code, open `localhost:{local-port}` on your browser.
