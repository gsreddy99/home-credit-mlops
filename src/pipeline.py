# filename: src/pipeline.py

import os
import argparse
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep
import sagemaker.processing


def get_pipeline(region: str, role: str, bucket: str) -> Pipeline:
    session = PipelineSession(default_bucket=bucket)

    # 1) Preprocess step (produces df_test.csv into silver bucket)
    preprocess = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
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
        ],
    )

    # 2) Evaluate step (writes evaluation.json into gold/evaluation)
    evaluate = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
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
            "--bucket", bucket,
        ],
    )

    # 3) Generate predictions step (writes df_subm.csv into gold/predictions)
    predictor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    step_predict = ProcessingStep(
        name="GeneratePredictions",
        processor=predictor,
        code="src/generate_predictions.py",
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="predictions",
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/home-credit/gold/predictions/"
            )
        ],
        job_arguments=[
            "--bucket", bucket,
        ],
    )

    return Pipeline(
        name="HomeCreditBatchPipeline",
        steps=[step_preprocess, step_evaluate, step_predict],
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
