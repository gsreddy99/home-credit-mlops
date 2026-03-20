# filename: src/pipeline.py

import os
import argparse
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep


def get_pipeline(region: str, role: str, bucket: str) -> Pipeline:
    session = PipelineSession(default_bucket=bucket)

    image_uri = (
        "683313688378.dkr.ecr.us-east-1.amazonaws.com/"
        "sagemaker-scikit-learn:1.2-1-cpu-py3"
    )

    ###########################################################################
    # PREPROCESS STEP
    ###########################################################################
    preprocess = ScriptProcessor(
        image_uri=image_uri,
        command=["bash"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=preprocess,
        code="bootstrap.sh",     # installs deps then runs preprocess.py
        source_dir="src",
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{bucket}/home-credit/silver/train/"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"s3://{bucket}/home-credit/silver/test/"
            ),
        ],
        job_arguments=[
            "preprocess.py",
            "--bucket", bucket,
            "--train-prefix", "bronze/train",
            "--test-prefix", "bronze/test",
        ],
    )

    ###########################################################################
    # EVALUATE STEP
    ###########################################################################
    evaluate = ScriptProcessor(
        image_uri=image_uri,
        command=["bash"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=evaluate,
        code="bootstrap.sh",
        source_dir="src",
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/home-credit/gold/evaluation/"
            )
        ],
        job_arguments=[
            "evaluate.py",
            "--output_dir", "/opt/ml/processing/evaluation"
        ],
    )

    ###########################################################################
    # FORCE SEQUENTIAL EXECUTION
    ###########################################################################
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
