# filename: src/pipeline.py
import sagemaker
from sagemaker.workflow.pipeline import Pipeline # Add this explicit import
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker import image_uris

def get_pipeline(region, role, bucket):
    # This session is tied to your specific bucket/region
    session = PipelineSession(default_bucket=bucket)

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
        sagemaker_session=session # Session used here
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=evaluate_processor,
        code="src/evaluate.py",
        inputs=[
            ProcessingInput(source="src/requirements.txt", destination="/opt/ml/processing/input/reqs")
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
        ]
    )

    # CRITICAL FIX: You must pass the 'session' to the Pipeline object.
    # If omitted, SageMaker tries to 'Create' a new session context
    # which often triggers the AccessDenied error in CodeBuild.
    return Pipeline(
        name="HomeCreditPipeline",
        steps=[step_evaluate],
        sagemaker_session=session
    )