###############################################
# Variables (assumed to be declared elsewhere)
# Make sure these exist in variables.tf or passed in:
# - var.bucket_name
# - var.codestar_connection_arn
###############################################

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
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:CreatePipeline",
          "sagemaker:UpdatePipeline",
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipelineExecution"
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
# CodeBuild Role & Policy
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

resource "aws_iam_policy" "codebuild_policy" {
  name = "homecredit-codebuild-policy-updated"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # ────────────────────────────────────────────────
      # S3 access for artifacts, sources, logs upload
      # ────────────────────────────────────────────────
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

      # ────────────────────────────────────────────────
      # CloudWatch Logs – this is the critical fix block
      # ────────────────────────────────────────────────
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

      # ────────────────────────────────────────────────
      # CodeBuild self-management (start, get status, etc.)
      # ────────────────────────────────────────────────
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

      # ────────────────────────────────────────────────
      # PassRole – required to pass SageMaker role to jobs
      # ────────────────────────────────────────────────
      {
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = aws_iam_role.sagemaker_execution_role.arn
      },

      # ────────────────────────────────────────────────
      # CodeStar Connections (GitHub / Bitbucket)
      # ────────────────────────────────────────────────
      {
        Effect = "Allow"
        Action = [
          "codeconnections:UseConnection",
          "codeconnections:GetConnectionToken",
          "codeconnections:GetConnection"
        ]
        Resource = var.codestar_connection_arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_codebuild_policy" {
  role       = aws_iam_role.homecredit_codebuild_role.name
  policy_arn = aws_iam_policy.codebuild_policy.arn
}