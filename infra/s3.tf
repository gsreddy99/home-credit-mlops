resource "aws_s3_bucket" "home_credit" {
  bucket = var.bucket_name
}

resource "aws_s3_object" "bronze_prefix" {
  bucket = aws_s3_bucket.home_credit.id
  key    = "home-credit/bronze/"
}

resource "aws_s3_object" "silver_prefix" {
  bucket = aws_s3_bucket.home_credit.id
  key    = "home-credit/silver/"
}

resource "aws_s3_object" "gold_prefix" {
  bucket = aws_s3_bucket.home_credit.id
  key    = "home-credit/gold/"
}
