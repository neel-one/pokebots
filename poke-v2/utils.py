import os
import json
import boto3

# To get the size of a S3 bucket
# aws s3 ls --summarize --human-readable --recursive s3://bucket-name/


class S3Client():
    """Thin wrapper over boto3.client('s3')
    Loads access key creds from ~/.aws/s3access.json
    which has format
    {
        "access_key": ...,
        "access_key_secret": ...
    }
    These access keys need permission to write/read from S3.
    """
    RAW_DATA_BUCKET = 'ps-random-raw-data'

    def __init__(self):
        with open(os.path.expanduser('~/.aws/s3access.json'), 'r') as f:
            s3access = json.load(f)
            self.s3_client = boto3.client('s3', 
                region_name='us-west-1', 
                aws_access_key_id=s3access['access_key'],
                aws_secret_access_key=s3access['access_key_secret'])
    
    def put_object(self, *args, **kwargs):
        self.s3_client.put_object(*args, **kwargs) 
    
    def download_logs(self, date, bucket=RAW_DATA_BUCKET):
        folder = date.strftime("%Y-%m-%d")
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=folder+'/'):
            print(page)

    def read_file(self, key, bucket=RAW_DATA_BUCKET):
        resp = self.s3_client.get_object(Bucket=bucket, Key=key)
        contents = resp['Body'].read()
        print(contents)
        return contents
