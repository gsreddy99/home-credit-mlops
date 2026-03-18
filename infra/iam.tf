###############################################
# SageMaker Execution Role
###############################################

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "homecredit-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
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
# CodeBuild Role
###############################################

resource "aws_iam_role" "homecredit_codebuild_role" {
  name = "homecredit-codebuild-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "codebuild.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "codebuild_policy" {
  name = "homecredit-codebuild-policy-updated"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 Access (kept your original)
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

      # CloudWatch Logs - scoped more tightly (optional belt-and-suspenders with managed policy)
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

      # CodeBuild self permissions
      {
        Effect = "Allow"
        Action = [
          "codebuild:BatchGetBuilds",
          "codebuild:StartBuild"
        ]
        Resource = "*"
      },

      # Allow CodeBuild to pass the SageMaker role
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = aws_iam_role.sagemaker_execution_role.arn
      },

      # GitHub/CodeStar connection (kept as-is)
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

# Attach your custom policy
resource "aws_iam_role_policy_attachment" "attach_codebuild_policy" {
  role       = aws_iam_role.homecredit_codebuild_role.name
  policy_arn = aws_iam_policy.codebuild_policy.arn
}

# Critical fix: Attach AWS managed policy for reliable CodeBuild + CloudWatch Logs permissions
resource "aws_iam_role_policy_attachment" "attach_codebuild_base_managed" {
  role       = aws_iam_role.homecredit_codebuild_role.name
  policy_arn = "arn:aws:iam::aws:policy/AWSCodeBuildBasePolicy" # Use this for minimal/reliable logging + build perms
  # Alternative (broader console access): "arn:aws:iam::aws:policy/AWSCodeBuildDeveloperAccess"
}