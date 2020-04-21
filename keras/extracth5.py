"""TensorFlow Script Mode 'training' job to import pre-trained Keras model(s)

About
=====

Importing pre-trained models for use in SageMaker endpoints and batch transformations requires
a basic understanding of:

1. The SageMaker API concept of a "Model", and the additional fields it requires beyond the
   artifact (e.g. container image URI);
2. The interface between your target SageMaker framework container (in this case TensorFlow) and
   the trained model artifact;
3. (If preparing the artifact on a notebook or similar), the installed dependencies on the notebook
   environment and how they relate to the target versions for training and inference.

This script; run as a dummy "training job" in SageMaker against an input consisting of one or more
.h5 Keras export files; simplifies the import of Keras models to a single- or multi-model SageMaker
TensorFlow Model.

A model artifact in SageMaker is a single `.tar.gz` archive which can be loaded into a container
image pre-built with appropriate dependencies and a framework application for model serving. A
deployable "Model" in SageMaker therefore requires both a container image URI and an artifact 
tarball.

Model Artifact Format
---------------------

Different frameworks (and, occasionally, different versions) expect different structures within the
tarball, and expose different methods for us to override the default behaviour. In TensorFlow's
case, TF Serving is configured to expect a particular folder structure containing `.pb` model
definition file(s) and accompanying `variables`.

Perhaps the best documentation for the tarball format is in the Python SDK TensorFlow package:

https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/deploying_tensorflow_serving.rst

(Particularly the section on "Deploying more than one model to your Endpoint")

Creating the SageMaker Model
----------------------------

By running this code in a SageMaker Training job, we simplify model creation:

* The version of TensorFlow is selected by the user (see "Usage" section below), without us messing
  with the version installed on the SageMaker Notebook Instance
* The container URI (which can be derived but is non-trivial to find, since AWS provides regional
  hosting and differently configured builds) is set for us by the SageMaker SDK
* All the other SageMaker Model registration is done for us by the SDK `fit()` function too, based
  on whatever parameters we provide to this dummy training job

For information on the Amazon SageMaker container URIs for TensorFlow serving, check the container
GitHub and the `create_image_uri()` function as used by the SageMaker SDK TensorFlow Estimator:

https://github.com/aws/sagemaker-tensorflow-serving-container#sagemaker-tensorflow-serving-container

https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/estimator.py


Usage
=====

Create and run the training job:

```
from sagemaker.tensorflow import TensorFlow as TensorFlowEstimator

estimator = TensorFlowEstimator(
    entry_point="extracth5.py",  # This filename
    source_dir="src",  # Parent folder, if more source files to be bundled

    base_job_name="keras-import",  # Prefix for the "training" job name
    output_path="s3://bucket/output-parent-folder"

    framework_version="1.15",  # Tested with 1.15
    script_mode=True,  # SageMaker supports two kinds of TF training - this job is script mode

    # We can run import on a cheap instance type, but be aware that by default the inference
    # instance type will be inferred from training, so GPU libs may be absent at inference.
    # ...you might want to use ml.p2.xlarge instead:
    train_instance_type="ml.m5.large",
    train_instance_count=1,  # Only one instance - this code isn't parallelized

    role=sagemaker.get_execution_role(),
)

estimator.fit({ "input": "s3://bucket/path/to/your/h5-file-or-parent-folder" })
```

...This shouldn't take long once the resource is provisioned, and will result in:

* Creation of one .tar.gz artifact (in a job-specific subfolder) under your `output_path` in S3
* Registration of a new "training job" in the SageMaker console

The SageMaker Model isn't quite created and registered yet, but can be done either:

* Explicitly using the `estimator.create_model()` function, or
* Implicitly by directly using `estimator.deploy()` to create a real-time endpoint or 
  `estimator.transformer()` for batch transformation.

https://sagemaker.readthedocs.io/en/stable/sagemaker.tensorflow.html#sagemaker.tensorflow.estimator.TensorFlow.create_model

https://sagemaker.readthedocs.io/en/stable/estimators.html#sagemaker.estimator.Estimator.deploy

https://sagemaker.readthedocs.io/en/stable/sagemaker.tensorflow.html#sagemaker.tensorflow.estimator.TensorFlow.transformer

If you don't use the Model straight away, you can always `attach()` the Estimator to the previous
result by training job name:

https://sagemaker.readthedocs.io/en/stable/estimators.html#sagemaker.estimator.Framework.attach


Citations and Further Reading
=============================

The model conversion code in this sample is taken from the following AWS Blog, which tackles the
task interactively instead of packaging it inside a SageMaker Training job:

https://aws.amazon.com/blogs/machine-learning/deploy-trained-keras-or-tensorflow-models-using-amazon-sagemaker/

"""

# FIXME: Multi-model output packages are being interpreted as single-model only for some reason

# Built-Ins:
import argparse
import os

# External Dependencies:
from keras import backend as K
import tensorflow as tf
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=os.environ.get('SM_CHANNEL_INPUT'),
        help="Folder on local file system where source model(s) are located"
    )
    parser.add_argument('--model-path', type=str, default=os.environ['SM_MODEL_DIR'],
        help="Folder on local file system where the model should be output"
    )

    # This seems to be a mandatory CLI arg for TensorFlow script mode, but we can ignore it:
    # (It's SageMaker, not us, that will upload the results from SM_MODEL_DIR to S3)
    parser.add_argument('--model_dir', type=str, default="",
        help="S3 URI where SageMaker will store the model (why is this an argument in SM TF!)"
    )

    args = parser.parse_args()
    if not args.input:
        raise argparse.ArgumentError(
            "h5 source folder must be specified via --input CLI or SM_CHANNEL_INPUT env variable"
        )

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    sourcefiles = list(filter(
        lambda s: s.lower().endswith(".h5"),
        os.listdir(args.input)
    ))

    n_models = len(sourcefiles)
    if not n_models:
        raise argparse.ArgumentError(f"No .h5 files found in input folder {args.input}")
    elif n_models > 1:
        print(f"{n_models} model files found: Creating bundled multi-model archive")
        model_ids = list(map(
            lambda s: s.rpartition(".")[0],  # Filename without extension is the ID
            sourcefiles
        ))
    else:
        print(f"One model file found: Creating single-model archive")
        model_ids = ["Servo"]  # The default ID when only one model is present

    for ix, sourcefile in enumerate(sourcefiles):
        print(f"Loading model {ix + 1} of {n_models}: {sourcefile}")
        model = tf.keras.models.load_model(
            f"{args.input}/{sourcefile}",
            # Implicitly, no custom_objects or etc
        )
        print("Model loaded")

        # Note we don't have logic here to handle multiple versions of the same model ID, so we
        # just set every model to version 1:
        export_path = f"{args.model_path}/export/{model_ids[ix]}/1"
        os.makedirs(export_path, exist_ok=True)

        builder = SavedModelBuilder(export_path)
        signature = predict_signature_def(
            inputs={ "inputs": model.input },
            outputs={ "score": model.output }
        )

        with K.get_session() as sess:
            # Save the meta graph and variables
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tag_constants.SERVING],
                signature_def_map={ "serving_default": signature }
            )
            builder.save()
            print("Model exported")
        K.clear_session()

    print("Done")
