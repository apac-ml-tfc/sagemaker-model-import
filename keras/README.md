# Keras Model Import

The [SageMaker TensorFlow Serving Container README](https://github.com/aws/sagemaker-tensorflow-serving-container#deploying-a-tensorflow-serving-model) has some great instructions on **Deploying a TensorFlow Serving Model**...

The good news is that in TFv2, TFServing can load tf.keras models automatically!

If you've got models from older versions, you'll need to convert them into TensorFlow format.

In this folder we provide a dummy "training" script for extracting (combined structure+weights) .h5 files into TF Serving compatible folders. You can either run it as a SageMaker "training" job for an Estimator you can `.deploy()`; or adapt the logic to create your own local `model.tar.gz` and follow the instructions linked above.
