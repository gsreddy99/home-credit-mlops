terraform {
  backend "s3" {
    bucket       = "sg-home-credit-tfstate"   # your backend bucket
    key          = "infra/terraform.tfstate"
    region       = "us-east-1"                # MUST match actual bucket region
    use_lockfile = true
  }
}
