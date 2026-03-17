terraform {
  backend "s3" {
    bucket       = "sg-home-credit-tfstate"
    key          = "infra/terraform.tfstate"
    region       = "us-east-1"
    use_lockfile = true
  }
}