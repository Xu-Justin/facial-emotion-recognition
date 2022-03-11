# facial-emotion-recognition

[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/Xu-Justin/facial-emotion-recognition)

The program will receive an image and recognize expression of detected faces in the images.

## Dataset

This project use dataset from Challenges in Representation Learning: Facial Expression Recognition Challenge (aka. FER2013). The dataset is available on [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

Run `generate_dataset.py` to download and extract the dataset.

## Docker

Docker images are available at [Docker Hub](https://hub.docker.com/repository/docker/jstnxu/facial-emotion-recognition/tags).

Read `README` file from the following links for more details on how to `build`, `pull`, or `run` the image.

 * [`face-detection:apps`](Apps/README.md)
 * [`face-detection:train`](train/README.md)

## References

[1] Khaireddin, Y., & Chen, Z. (2021). Facial emotion recognition: State of the art performance on FER2013. arXiv preprint arXiv:2105.03588.

[2] Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ... & Bengio, Y. (2013, November). Challenges in representation learning: A report on three machine learning contests. In International conference on neural information processing (pp. 117-124). Springer, Berlin, Heidelberg.

---

This project was developed as part of Nodeflux Internship x Kampus Merdeka.
