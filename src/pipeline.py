# filename: src/pipeline.py
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker import image_uris

def get_pipeline(region, role, bucket):
    session = PipelineSession(default_bucket=bucket)

    # This image allows the NumPy 2.0 upgrade without crashing
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
            ProcessingInput(source="src/requirements.txt", destination="/opt/ml/processing/input/reqs")
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
        ]
    )
    return sagemaker.workflow.pipeline.Pipeline(name="HomeCreditPipeline", steps=[step_evaluate])