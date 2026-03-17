variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "bucket_name" {
  type    = string
  default = "sg-home-credit"
}

variable "github_owner" {
  type = string
}

variable "github_repo" {
  type    = string
  default = "home-credit-mlops"
}

variable "codestar_connection_arn" {
  type        = string
  description = "AWS CodeStar connection ARN for GitHub"
}
