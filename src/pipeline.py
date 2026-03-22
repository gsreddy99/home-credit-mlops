# filename: src/pipeline.py

import os
import argparse
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker import image_uris


def get_pipeline(region, role, bucket):
    session = PipelineSession(default_bucket=bucket)

    # Use the same sklearn container for all steps
    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        instance_type="ml.m5.2xlarge"
    )

    # ============================================================
    # 1) PREPROCESS STEP
    # ============================================================
    preprocess_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=preprocess_processor,
        code="src/preprocess.py",
        job_arguments=[
            "--bucket", bucket,
            "--train-prefix", "home-credit/bronze/train",
            "--test-prefix", "home-credit/bronze/test",
        ],
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
    )

    # ============================================================
    # 2) TRAIN STEP
    # ============================================================
    train_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step_train = ProcessingStep(
        name="TrainModel",
        processor=train_processor,
        code="src/train.py",
        job_arguments=[
            "--model_output", "/opt/ml/processing/model"
        ],
        inputs=[
            ProcessingInput(
                source="src/requirements.txt",
                destination="/opt/ml/processing/input/reqs"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/model",
                destination=f"s3://{bucket}/home-credit/model/"
            )
        ],
    )

    # Train depends on Preprocess
    step_train.add_depends_on([step_preprocess])

    # ============================================================
    # 3) EVALUATE STEP
    # ============================================================
    evaluate_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=evaluate_processor,
        code="src/evaluate.py",
        inputs=[
            ProcessingInput(
                source="src/requirements.txt",
                destination="/opt/ml/processing/input/reqs"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/home-credit/gold/evaluation/"
            )
        ],
    )

    # Evaluate depends on Train
    step_evaluate.add_depends_on([step_train])

    # ============================================================
    # PIPELINE
    # ============================================================
    return Pipeline(
        name="HomeCreditBatchPipeline",
        steps=[step_preprocess, step_train, step_evaluate],
        sagemaker_session=session,
    )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    region = os.environ.get("AWS_REGION", "us-east-1")
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    bucket = os.environ.get("BUCKET")

    pipeline = get_pipeline(region, role, bucket)

    if args.update:
        print(f"Upserting pipeline: {pipeline.name}")
        pipeline.upsert(role_arn=role)

    if args.run:
        print(f"Starting execution for: {pipeline.name}")
        execution = pipeline.start()
        print(f"Execution started! ARN: {execution.arn}")
