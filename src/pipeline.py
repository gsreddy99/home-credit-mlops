# filename: src/pipeline.py

import os
import argparse
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep
import sagemaker.processing
from sagemaker import image_uris


def get_pipeline(region: str, role: str, bucket: str) -> Pipeline:
    session = PipelineSession(default_bucket=bucket)

    # UPDATED IMAGE URI (modern container with NumPy 1.26+)
    # Option 1: Preferred for XGBoost (small, has lightgbm in many versions, good for your use case)
    # Replace the existing image_uri block with:
    image_uri = sagemaker.image_uris.retrieve(
        framework="lightgbm",          # ← this is the key change
        region=region,
        version="3.3-1",               # Recent stable version; check docs or try "latest" / "3.4-1" if available
        instance_type="ml.m5.2xlarge",
        image_scope="training"         # or omit — "training" works for processing too
    )


    ###########################################################################
    # 1) PREPROCESS STEP
    ###########################################################################
    preprocess = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type="ml.m5.2xlarge",
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
        job_arguments=[
            "--bucket", bucket,
            "--train-prefix", "home-credit/bronze/train",
            "--test-prefix", "home-credit/bronze/test",
        ],
    )

    ###########################################################################
    # 2) EVALUATE STEP
    ###########################################################################
    evaluate = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type="ml.m5.2xlarge",
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
        job_arguments=[
            "--output_dir", "/opt/ml/processing/evaluation"
        ],
    )

    step_evaluate.add_depends_on([step_preprocess])

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
