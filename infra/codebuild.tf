resource "aws_codebuild_project" "homecredit" {
  name          = "homecredit-batch-pipeline"
  description   = "Runs SageMaker batch pipeline for Home Credit"
  service_role  = aws_iam_role.homecredit_codebuild_role.arn
  build_timeout = 60

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_SMALL"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = false

    environment_variable {
      name  = "AWS_REGION"
      value = var.aws_region
    }

    environment_variable {
      name  = "SAGEMAKER_ROLE_ARN"
      value = aws_iam_role.sagemaker_execution_role.arn
    }

    environment_variable {
      name  = "BUCKET"
      value = var.bucket_name
    }
  }

  source {
    type            = "GITHUB"
    location        = "https://github.com/${var.github_owner}/${var.github_repo}.git"
    git_clone_depth = 1
    buildspec       = "buildspec.yml"

    auth {
      type     = "CODECONNECTIONS"
      resource = var.codestar_connection_arn
    }
  }

  logs_config {
    cloudwatch_logs {
      status = "ENABLED"
    }
  }
}
