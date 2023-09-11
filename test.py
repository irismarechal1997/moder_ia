from google.oauth2 import service_account
import os
from google.cloud import storage
import pandas as pd

credential=service_account.Credentials.from_service_account_file(os.environ["GCP_PATH"])
print(credential)
client = storage.Client(project=os.environ["GCP_PROJECT"], credentials=credential)
bucket = client.bucket("testdjflajdnclkdsjnclisud")

## test upload
# blob = bucket.blob("test.py")
# blob.upload_from_filename("test.py")
# pd.read_csv("gs://testdjflajdnclkdsjnclisud/submission_baseline")

## test download
blob = bucket.blob("submission_baseline.csv")
blob.download_to_filename("submission_baseline.csv")
