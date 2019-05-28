import boto3
import os

def upload_to_s3(source, bucket, key):
    s3 = boto3.client('s3')
    s3.meta.client.upload_file(source, bucket, key)

def download_from_s3(key, bucket, target):
    # Make sure directory exists before downloading to it.
    target_dir = os.path.dirname(target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    s3 = boto3.client('s3')
    try:
      s3.download_file(bucket, key, target)
    except Exception as e:
      print(e)
