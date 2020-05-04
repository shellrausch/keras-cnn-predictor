# Binary prediction with Keras on CNNs

## Usage

Build the image

```shell
docker build -t prediction .
```

Run the image

```shell
docker run --rm                                         \
    -v <YOUR_DIR_WITH_IMAGES>:/prediction/tmp           \
    -v <YOUR_DIR_WITH_MODELS>:/prediction/models        \
    -e MODEL_NAME=<YOUR_MODEL_NAME_FROM_MODEL_DIR>.h5   \
    prediction
```
