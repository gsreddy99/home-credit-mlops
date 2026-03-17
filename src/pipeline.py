import os
import argparse
import sagemaker
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.inputs import TrainingInput


def get_pipeline(
    region,
    role,
    bucket,
    base_prefix="home-credit",
):
    """
    Creates a SageMaker Pipeline with:
    1. Preprocessing step
    2. Training step
    3. Evaluation step
    """

    pipeline_session = PipelineSession()

    # Pipeline parameters
    input_bucket = ParameterString(name="InputBucket", default_value=bucket)
    base_prefix_param = ParameterString(name="BasePrefix", default_value=base_prefix)

    # Cache config (optional)
    cache_config = CacheConfig(enable_caching=False)

    # Step 1: Preprocessing
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="preprocess",
        role=role,
        sagemaker_session=pipeline_session,
    )

    step_preprocess = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{bucket}/{base_prefix}/silver/train/",
            ),
            sagemaker.processing.ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"s3://{bucket}/{base_prefix}/silver/test/",
            ),
        ],
        code="src/preprocess.py",
        cache_config=cache_config,
    )

    # Step 2: Training
    estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.7-1",
        ),
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
        output_path=f"s3://{bucket}/{base_prefix}/models/",
    )

    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=f"s3://{bucket}/{base_prefix}/silver/train/"
            )
        },
        cache_config=cache_config,
    )

    # Step 3: Evaluation
    sklearn_eval = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="evaluate",
        role=role,
        sagemaker_session=pipeline_session,
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=sklearn_eval,
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/{base_prefix}/gold/evaluation/",
            )
        ],
        code="src/evaluate.py",
        cache_config=cache_config,
    )

    # Build pipeline
    pipeline = Pipeline(
        name="HomeCreditPipeline",
        parameters=[input_bucket, base_prefix_param],
        steps=[step_preprocess, step_train, step_evaluate],
        sagemaker_session=pipeline_session,
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    region = os.environ.get("AWS_REGION", "us-east-1")
    role = os.environ["SAGEMAKER_ROLE_ARN"]
    bucket = os.environ["BUCKET"]

    pipeline = get_pipeline(region=region, role=role, bucket=bucket)

    if args.update:
        print("Updating/Creating SageMaker Pipeline...")
        pipeline.upsert(role_arn=role)
        print("Pipeline updated.")

    if args.run:
        print("Starting Pipeline Execution...")
        execution = pipeline.start()
        print("Execution ARN:", execution.arn)


if __name__ == "__main__":
    main()
