# filename: src/pipeline.py
import os
import argparse
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker import image_uris

def get_pipeline(region, role, bucket):
    session = PipelineSession(default_bucket=bucket)

    # Use the Scikit-Learn image to allow NumPy 2.x upgrades
    eval_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        instance_type="ml.m5.2xlarge"
    )

    evaluate_processor = ScriptProcessor(
        image_uri=eval_image,
        command=["python3"],
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        role=role,
        sagemaker_session=session
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=evaluate_processor,
        code="src/evaluate.py",
        inputs=[
            # Ensure requirements.txt is passed to the container
            ProcessingInput(source="src/requirements.txt", destination="/opt/ml/processing/input/reqs")
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
        ]
    )

    return Pipeline(
        name="HomeCreditPipeline",
        steps=[step_evaluate],
        sagemaker_session=session
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    # These environment variables must be set in your CodeBuild project
    region = os.environ.get("AWS_REGION", "us-east-1")
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    bucket = os.environ.get("BUCKET", "sg-home-credit")

    pipeline = get_pipeline(region, role, bucket)

    if args.update:
        print(f"Updating pipeline definition: {pipeline.name}")
        pipeline.upsert(role_arn=role)

    if args.run:
        print(f"Starting pipeline execution...")
        execution = pipeline.start()
        print(f"Execution started: {execution.arn}")