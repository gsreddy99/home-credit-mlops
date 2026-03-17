import os
import argparse
import sagemaker
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline_context import PipelineSession
import sagemaker.processing


def get_pipeline(region, role, bucket):

    # Force SageMaker to use YOUR bucket instead of creating a default one
    session = PipelineSession(default_bucket=bucket)

    # -------------------------
    # PREPROCESS STEP (FREE TIER)
    # -------------------------
    preprocess = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.t3.micro",     # FREE TIER
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=preprocess,
        code="src/preprocess.py",
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
    )

    # -------------------------
    # EVALUATE STEP (FREE TIER)
    # -------------------------
    evaluate = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.t3.micro",     # FREE TIER
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=evaluate,
        code="src/evaluate.py",
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/home-credit/gold/evaluation/"
            )
        ],
    )

    # -------------------------
    # PIPELINE DEFINITION
    # -------------------------
    return Pipeline(
        name="HomeCreditPipeline",
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
