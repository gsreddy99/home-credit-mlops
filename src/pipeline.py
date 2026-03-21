# filename: src/pipeline.py
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

    # INFRASTRUCTURE FIX: Use Scikit-Learn image for the Evaluate step.
    # The XGBoost image is version-locked and breaks when you upgrade NumPy.
    eval_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        instance_type="ml.m5.2xlarge",
    )

    ###########################################################################
    # 1) PREPROCESS STEP (Keep existing XGBoost if you prefer for this step)
    ###########################################################################
    # ... (Your existing preprocess logic) ...

    ###########################################################################
    # 2) EVALUATE STEP
    ###########################################################################
    evaluate_processor = ScriptProcessor(
        image_uri=eval_image,
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
            # CRITICAL: This maps your requirements file so evaluate.py can see it
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
        job_arguments=["--output_dir", "/opt/ml/processing/evaluation"],
    )

    return Pipeline(
        name="HomeCreditBatchPipeline",
        steps=[step_evaluate], # Add step_preprocess here if needed
        sagemaker_session=session,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    region = os.environ.get("AWS_REGION")
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    bucket = os.environ.get("BUCKET")

    if not all([region, role, bucket]):
        raise ValueError(f"Missing env vars. Region: {region}, Role: {role}, Bucket: {bucket}")

    pipeline = get_pipeline(region, role, bucket)

    if args.update:
        print("Upserting pipeline...")
        pipeline.upsert(role_arn=role)

    if args.run:
        print("Starting execution...")
        execution = pipeline.start()
        print(f"Execution started: {execution.arn}")

if __name__ == "__main__":
    main()