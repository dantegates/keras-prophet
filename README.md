# Batch Prophet

![CI](https://github.com/dantegates/keras-prophet/actions/workflows/python-app.yml/badge.svg)

Like [Facebook's Prophet](https://facebook.github.io/prophet/) but with more
flexibility. Supporting

- Batch predictions: the ability to fit to many series at once
- Global and Local seasonal/trend features
- Arbitrary loss functions
- Integration with [tensorflow](https://www.tensorflow.org/) for unlimited extensibility
