# filename: src/pipeline.py
import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker import image_uris

def get_pipeline(region, role, bucket):
    session = PipelineSession(default_bucket=bucket)

    # INFRASTRUCTURE FIX: Use Sklearn instead of XGBoost for the Eval step
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
            # Ensure requirements.txt is passed to the processor
            ProcessingInput(source="src/requirements.txt", destination="/opt/ml/processing/input/reqs")
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation",
                             destination=f"s3://{bucket}/home-credit/gold/evaluation/")
        ]
    )

    return Pipeline(name="HomeCreditBatchPipeline", steps=[step_evaluate], sagemaker_session=session)

if __name__ == "__main__":
    region, role, bucket = os.environ.get("AWS_REGION"), os.environ.get("SAGEMAKER_ROLE_ARN"), os.environ.get("BUCKET")
    pipeline = get_pipeline(region, role, bucket)
    pipeline.upsert(role_arn=role)