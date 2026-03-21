import os
import argparse
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker import image_uris

def get_pipeline(region: str, role: str, bucket: str) -> Pipeline:
    session = PipelineSession(default_bucket=bucket)

    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1",
        instance_type="ml.m5.2xlarge",
    )

    # 1) PREPROCESS STEP (Simplified for brevity)
    preprocess = ScriptProcessor(
        image_uri=image_uri, command=["python3"], instance_type="ml.m5.2xlarge",
        instance_count=1, role=role, sagemaker_session=session,
    )

    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=preprocess,
        code="src/preprocess.py",
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train",
                             destination=f"s3://{bucket}/home-credit/silver/train/"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test",
                             destination=f"s3://{bucket}/home-credit/silver/test/"),
        ],
        job_arguments=["--bucket", bucket, "--train-prefix", "home-credit/bronze/train", "--test-prefix", "home-credit/bronze/test"],
    )

    # 2) EVALUATE STEP
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
        inputs=[
            # Map requirements.txt from your local src/ folder to the container
            ProcessingInput(source="src/requirements.txt", destination="/opt/ml/processing/input/reqs")
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation",
                             destination=f"s3://{bucket}/home-credit/gold/evaluation/")
        ],
        job_arguments=["--output_dir", "/opt/ml/processing/evaluation"],
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

    region, role, bucket = os.environ.get("AWS_REGION"), os.environ.get("SAGEMAKER_ROLE_ARN"), os.environ.get("BUCKET")
    if not all([region, role, bucket]): raise ValueError("Missing env vars")

    pipeline = get_pipeline(region, role, bucket)

    if args.update: pipeline.upsert(role_arn=role)
    if args.run: pipeline.start()

if __name__ == "__main__":
    main()