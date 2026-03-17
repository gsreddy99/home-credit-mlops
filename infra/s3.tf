resource "aws_s3_bucket" "home_credit" {
  bucket = "sg-home-credit"

  tags = {
    Name = "sg-home-credit"
  }
}

resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.home_credit.id

  versioning_configuration {
    status = "Enabled"
  }
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

resource "aws_s3_object" "bronze_train_prefix" {
  bucket = aws_s3_bucket.home_credit.id
  key    = "home-credit/bronze/train/"
}

resource "aws_s3_object" "bronze_test_prefix" {
  bucket = aws_s3_bucket.home_credit.id
  key    = "home-credit/bronze/test/"
}
