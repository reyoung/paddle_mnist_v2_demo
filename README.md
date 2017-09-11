## Paddle MNIST DEMO

* Build training image
  * `docker build -f Dockerfile.train . -t paddle_mnist`
* Run training image
  * `docker run --rm -v $SOMEWHERE_SAME_MODEL:/output paddle_mnist`
