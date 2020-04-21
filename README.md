# SageMaker Model Import

Can I bring models already trained elsewhere into SageMaker for real-time deployment or batch transform? **Absolutely!**

...But most SageMaker tutorials cover end-to-end model building within the platform using the high level `Estimator` / `Estimator.deploy()` interface provided by the [SageMaker SDK](https://sagemaker.readthedocs.io/en/stable/index.html).

This high-level interface is nice, but it abstracts away the underlying steps you might need to think about for importing your own model: Skimming over the creation of the [Model](https://sagemaker.readthedocs.io/en/stable/model.html) and [Endpoint Configuration](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpointConfig.html) objects.

There are some nuances depending on the modelling framework you're using and what kind of I/O format you need to support, but it essentially boils down to:

1. Supply a SageMaker-compatible container image URI that your model will run in
  * This will typically be one of the pre-built containers by SageMaker
  * ...Or it could be a custom image in Amazon ECR if you have a specific need
2. Understand the contract between your target container image and the model artifact
3. Create a `model.tar.gz` model artifact in S3 (e.g. containing your neural network structure+weights, etc.)
4. Register the SageMaker "Model" from this artifact, your container image URI, and any other parameters you need
  * Models can be created in UI through the ["Models" tab of the SageMaker Console](https://console.aws.amazon.com/sagemaker/home#/models)
  * ...Or via the [Model class](https://sagemaker.readthedocs.io/en/stable/model.html) of the [SageMaker SDK](https://sagemaker.readthedocs.io/en/stable/index.html)
  * ...Or (if you must) via low-level CLI/API tools through the [CreateModel API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateModel.html)

Once your SageMaker "Model" has been created, deploying it to a real-time endpoint (or creating a batch transform job) should be pretty straightforward (through the console UI, or SDKs/APIs)

## Hints and Tips

### Does my framework have a pre-built container?

Probably... see the [list of frameworks](https://docs.aws.amazon.com/sagemaker/latest/dg/frameworks.html) in the [Amazon SageMaker docs](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) for more info!

**Note: Keras** started out as a framework-agnostic API, but is now a core TensorFlow component. Chances are, if you're importing Keras models you want the TensorFlow container.

### How do I find the container URI?

OK, this part's not as easy as it could be (see the "A Handy Cheat" section below for an alternative!)... In general, you're probably best finding the public GitHub for your framework's serving container. For example

* [aws/sagemaker-pytorch-serving-container](https://github.com/aws/sagemaker-pytorch-serving-container)
* [aws/sagemaker-tensorflow-serving-container](https://github.com/aws/sagemaker-tensorflow-serving-container)
* ...etc.

ECR container URIs look like this: (variables in CAPS)

```
AWS_ACCOUNT_ID.dkr.ecr.AWS_REGION.amazonaws.com/REPOSITORY_NAME:TAG
```

Typically the TAG combines several important factors (e.g. framework version, Python version, and whether GPU support is built in) - so you need to refer to the README and not just try `:latest`.

You need to use the AWS region you're deploying into, but the AWS account ID where AWS is hosting the repository... Bear in mind that different regions/versions might be hosted on different account IDs.

Some examples at the time of writing:

```
520713654638.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tensorflow-serving:1.12-gpu-py3
763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/tensorflow-inference:2.0.0-cpu-py3
```

### What do I need in my model.tar.gz?

The answer to this really depends on what framework you're targeting...

SageMaker training jobs just zip and upload the final contents of `/opt/ml/model`, so if the serving container GitHub isn't giving spec details - check out the model training instructions in the [SageMaker SDK docs](https://sagemaker.readthedocs.io/) for your framework - particularly what your script should save.


## An interesting cheat: Dummy training jobs

Sick of hunting for a container URI? Don't want the hassle of tarballing and registering your `Model` in SageMaker? Why not let the SageMaker SDK do it for you!

* Upload your existing model artifacts to S3
* Create a dummy "training job" script that simply copies the contents of folder `os.environ["SM_CHANNEL_INPUT"]` to `os.environ["SM_MODEL_DIR"]`
* Create the SageMaker `Estimator` for your target framework, pointing at your script
* Call `estimator.fit({ "input": "s3://bucket-name/path/to/your/existing/model" })` to "train" the estimator

The `estimator` SDK object knows where to find the container image for the version you requested, and the "training job" tarballed your artifacts into a `model.tar.gz` on S3 - so now you'll be able to call `estimator.deploy()` to create an endpoint, as usual!

This method is a bit of a strange way of doing things, but might be useful for folks who like the SageMaker `Estimator` tooling - because there are so many more code examples out there for `estimator.fit()` / `estimator.deploy()` than using `Model`s.
