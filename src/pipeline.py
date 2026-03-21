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

    # Use XGBoost container (small, fast, reliable URI retrieval)
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1",                   # stable version; "1.5-1" or "latest" also fine
        instance_type="ml.m5.2xlarge",
    )

    print(f"Using processing image: {image_uri}")

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
        dependencies=["src/requirements.txt"],  # ← installs lightgbm
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
        dependencies=["src/requirements.txt"],  # ← installs lightgbm
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
    parser.add_argument("--update", action="store_true", help="Upsert the pipeline definition")
    parser.add_argument("--run", action="store_true", help="Start a pipeline execution")
    args = parser.parse_args()

    region = os.environ.get("AWS_REGION")
    role   = os.environ.get("SAGEMAKER_ROLE_ARN")
    bucket = os.environ.get("BUCKET")

    if not all([region, role, bucket]):
        missing = [k for k, v in {"AWS_REGION": region, "SAGEMAKER_ROLE_ARN": role, "BUCKET": bucket}.items() if not v]
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    pipeline = get_pipeline(region, role, bucket)

    if args.update:
        print("Upserting SageMaker pipeline...")
        pipeline.upsert(role_arn=role)
        print("Pipeline definition upserted successfully.")

    if args.run:
        print("Starting pipeline execution...")
        execution = pipeline.start()
        print(f"Pipeline execution started. Execution ARN: {execution.arn}")


if __name__ == "__main__":
    main()