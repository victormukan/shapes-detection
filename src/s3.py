import boto3
from botocore.exceptions import NoCredentialsError
import os

ACCESS_KEY = os.environ.get("AWS_KEY")
SECRET_KEY = os.environ.get("AWS_SECRET")
BUCKET = os.environ.get("AWS_BUCKET")

def upload_to_aws(local_file, s3_file):

  s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

  try:
    s3.upload_file(local_file, BUCKET, s3_file)
    print("Upload Successful")
    return True
  except FileNotFoundError:
    print("The file was not found")
    return False
  except NoCredentialsError:
    print("Credentials not available")
    return False
