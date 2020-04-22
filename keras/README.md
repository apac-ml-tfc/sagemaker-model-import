# Keras Model Import

Check out this [official AWS blog post](https://aws.amazon.com/blogs/machine-learning/deploy-trained-keras-or-tensorflow-models-using-amazon-sagemaker/) for a great guide on deploying existing Keras models to SageMaker - including step-by-step code examples!

The good news is that in TFv2, TFServing can load tf.keras models automatically! If you've got models from older versions, you'll need to convert them into TensorFlow format (as the blog post does).

For detailed guidance on the container and framework application, see the [SageMaker TensorFlow Serving Container README](https://github.com/aws/sagemaker-tensorflow-serving-container#deploying-a-tensorflow-serving-model) too.

The [TensorFlowModel](https://sagemaker.readthedocs.io/en/stable/sagemaker.tensorflow.html#tensorflow-model) class in the [SageMaker Python SDK](https://sagemaker.readthedocs.io/) can simplify the process of creating a SageMaker `Model` from your `model.tar.gz` artifact, by automatically configuring the container image URI based on the TensorFlow and Python version parameters you provide.

In this folder we provide a dummy "training" script for extracting (combined structure+weights) .h5 files into TF Serving compatible folders. You can either run it as a SageMaker "training" job for an Estimator you can `.deploy()`; or adapt the logic to create your own local `model.tar.gz` and follow the instructions linked above.
