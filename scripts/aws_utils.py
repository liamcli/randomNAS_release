import boto3

def upload_to_s3(source, bucket, key):
    s3 = boto3.client('s3')
    s3.meta.client.upload_file(source, bucket, key)

def download_from_s3(key, bucket, target):
    s3 = boto3.client('s3')
    try:
      s3.download_file(bucket, key, target)
    except Exception as e:
      print(e)
