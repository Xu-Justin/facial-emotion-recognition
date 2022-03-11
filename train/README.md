# `facial-emotion-recognition:train`

## Build Image

```
$ docker build -t jstnxu/facial-emotion-recognition:train .
```

## Pull Image from Docker Hub

```
$ docker pull jstnxu/facial-emotion-recognition:train
```

## Run Image

To run this image, run the following code. Change `{local-directory}` with absolute path of your local directory.

```
$ docker run --gpus all -v {local-directory}:/app/local jstnxu/facial-emotion-recognition:train \
    --EPOCHS int \
    --BATCH_SIZE int \
    --PATH_TRAIN_DATASET path \
    --PATH_EVAL_DATASET path \
    [--PATH_MODEL_WEIGHT path] \
    --PATH_OUTPUT_MODEL_WEIGHT path \
    [--CPU_ONLY] \
    [--CLASSIFIER_ONLY]
```

### Example

```
docker run --gpus all -v /home/justin-xu/Nodeflux/facial-emotion-recognition:/app/local jstnxu/facial-emotion-recognition:train \
    --EPOCHS 2 \
    --BATCH_SIZE 8 \
    --PATH_TRAIN_DATASET ./local/dataset/fer2013/train/ \
    --PATH_EVAL_DATASET ./local/dataset/fer2013/val/ \
    --PATH_MODEL_WEIGHT ./local/weight.zip \
    --PATH_OUTPUT_MODEL_WEIGHT ./local/new_weight.zip \
    --CLASSIFIER_ONLY
```
