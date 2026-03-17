resource "aws_codebuild_project" "homecredit" {
  name         = "homecredit-build"
  service_role = aws_iam_role.codebuild_role.arn

  environment {
    compute_type = "BUILD_GENERAL1_SMALL"
    image        = "aws/codebuild/standard:7.0"
    type         = "LINUX_CONTAINER"

    environment_variable {
      name  = "BUCKET"
      value = var.bucket_name
    }

    environment_variable {
      name  = "SAGEMAKER_ROLE_ARN"
      value = aws_iam_role.sagemaker_execution.arn
    }
  }

  source {
    type     = "GITHUB"
    location = "https://github.com/${var.github_owner}/${var.github_repo}.git"
  }

  artifacts {
    type = "NO_ARTIFACTS"
  }
}
