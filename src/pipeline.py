# filename: src/pipeline.py

import os
import argparse
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep
import sagemaker.processing


def get_pipeline(region: str, role: str, bucket: str) -> Pipeline:
    session = PipelineSession(default_bucket=bucket)

    ###########################################################################
    # 1) PREPROCESS STEP
    ###########################################################################
    preprocess = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=preprocess,
        code="preprocess.py",
        source_dir="src",                     # ensures requirements.txt is included
        dependencies=["requirements.txt"],    # installs polars, lightgbm, sklearn, etc.
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{bucket}/home-credit/silver/train/"
            ),
            sagemaker.processing.ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"s3://{bucket}/home-credit/silver/test/"
            ),
        ],
        job_arguments=[
            "--bucket", bucket,
            "--train-prefix", "bronze/train",
            "--test-prefix", "bronze/test",
        ],
    )

    ###########################################################################
    # 2) EVALUATE STEP (uses existing model aiml_model.pkl)
    ###########################################################################
    evaluate = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=evaluate,
        code="evaluate.py",
        source_dir="src",
        dependencies=["requirements.txt"],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/home-credit/gold/evaluation/"
            )
        ],
        job_arguments=[
            "--output_dir", "/opt/ml/processing/evaluation"
        ],
    )

    ###########################################################################
    # FORCE STRICT SEQUENTIAL EXECUTION
    ###########################################################################
    step_evaluate.add_depends_on([step_preprocess])

    ###########################################################################
    # PIPELINE: Preprocess → Evaluate
    ###########################################################################
    return Pipeline(
        name="HomeCreditBatchPipeline",
        steps=[step_preprocess, step_evaluate],
        sagemaker_session=session,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    region = os.environ["AWS_REGION"]
    role = os.environ["SAGEMAKER_ROLE_ARN"]
    bucket = os.environ["BUCKET"]

    pipeline = get_pipeline(region, role, bucket)

    if args.update:
        pipeline.upsert(role_arn=role)

    if args.run:
        execution = pipeline.start()
        print("Execution ARN:", execution.arn)


if __name__ == "__main__":
    main()
