output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker_execution_role.arn
}

output "codebuild_role_arn" {
  value = aws_iam_role.homecredit_codebuild_role.arn
}
