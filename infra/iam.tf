###############################################
# SageMaker Execution Role
###############################################

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "homecredit-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "sagemaker_policy" {
  name = "homecredit-sagemaker-batch-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [

      ############################################################
      # S3 access for pipeline inputs/outputs
      ############################################################
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.bucket_name}",
          "arn:aws:s3:::${var.bucket_name}/*"
        ]
      },

      ############################################################
      # CloudWatch logs for processing jobs
      ############################################################
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },

      ############################################################
      # SageMaker processing + pipeline operations
      ############################################################
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:CreatePipeline",
          "sagemaker:UpdatePipeline",
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipelineExecution",
          "sagemaker:DescribePipeline"
        ]
        Resource = "*"
      },

      ############################################################
      # REQUIRED FIX — Tagging permissions for pipeline jobs
      ############################################################
      {
        Effect = "Allow"
        Action = [
          "sagemaker:AddTags",
          "sagemaker:ListTags"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_sagemaker_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_policy.arn
}

###############################################
# CodeBuild Role
###############################################

resource "aws_iam_role" "homecredit_codebuild_role" {
  name = "homecredit-codebuild-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codebuild.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

###############################################
# CodeBuild Policy (FINAL, COMPLETE)
###############################################

resource "aws_iam_policy" "codebuild_policy" {
  name = "homecredit-codebuild-policy-updated"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [

      ############################################################
      # S3 access for source, artifacts, and pipeline data
      ############################################################
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:PutObject",
          "s3:PutObjectAcl",
          "s3:ListBucket",
          "s3:GetBucketAcl",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:aws:s3:::${var.bucket_name}",
          "arn:aws:s3:::${var.bucket_name}/*"
        ]
      },

      ############################################################
      # CloudWatch Logs for CodeBuild
      ############################################################
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          "arn:aws:logs:*:*:log-group:/aws/codebuild/*",
          "arn:aws:logs:*:*:log-group:/aws/codebuild/*:*"
        ]
      },

      ############################################################
      # CodeBuild self-management
      ############################################################
      {
        Effect = "Allow"
        Action = [
          "codebuild:BatchGetBuilds",
          "codebuild:StartBuild",
          "codebuild:StopBuild",
          "codebuild:GetResourcePolicy"
        ]
        Resource = "*"
      },

      ############################################################
      # PassRole to SageMaker execution role
      ############################################################
      {
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = aws_iam_role.sagemaker_execution_role.arn
      },

      ############################################################
      # CodeStar Connections (GitHub)
      ############################################################
      {
        Effect = "Allow"
        Action = [
          "codeconnections:UseConnection",
          "codeconnections:GetConnectionToken",
          "codeconnections:GetConnection"
        ]
        Resource = var.codestar_connection_arn
      },

      ############################################################
      # SageMaker Pipeline Permissions
      ############################################################
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreatePipeline",
          "sagemaker:UpdatePipeline",
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipeline",
          "sagemaker:DescribePipelineExecution"
        ]
        Resource = "arn:aws:sagemaker:us-east-1:943938400093:pipeline/HomeCreditBatchPipeline"
      },

      ############################################################
      # List operations must be wildcard
      ############################################################
      {
        Effect = "Allow"
        Action = [
          "sagemaker:ListPipelines",
          "sagemaker:ListPipelineExecutions"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_codebuild_policy" {
  role       = aws_iam_role.homecredit_codebuild_role.name
  policy_arn = aws_iam_policy.codebuild_policy.arn
}
