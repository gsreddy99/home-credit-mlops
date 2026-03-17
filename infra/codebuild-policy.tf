resource "aws_iam_role_policy" "codebuild_policy" {
  name = "homecredit-codebuild-policy"
  role = aws_iam_role.homecredit_codebuild_role.id

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
          "arn:aws:s3:::sg-home-credit",
          "arn:aws:s3:::sg-home-credit/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreatePipeline",
          "sagemaker:UpdatePipeline",
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipeline",
          "sagemaker:DescribePipelineExecution",
          "sagemaker:ListPipelineExecutions"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = "arn:aws:iam::943938400093:role/homecredit-sagemaker-execution"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}
