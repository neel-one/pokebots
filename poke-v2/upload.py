import requests
from multiprocessing import Pool
import time
from utils import S3Client

s3_client = S3Client()

def upload_url_contents_to_s3(url_bucket_tuple):
    """
    Uploads content from a URL to an Amazon S3 bucket. Takes a tuple containing the URL and the bucket info.

    Args:
    - url_bucket_tuple (tuple): Tuple containing the URL, bucket name, and S3 object name.
    """
    url, bucket_name, s3_object_name = url_bucket_tuple
    
    response = requests.get(url)
    if response.status_code == 200:
        try:
            s3_client.put_object(Bucket=bucket_name, Key=s3_object_name, Body=response.text)
            print(f"File uploaded successfully to {bucket_name}/{s3_object_name}")
        except Exception as e:
            print(f"An error occurred uploading to S3: {e}")
    else:
        print(f"Failed to retrieve data from URL: {url} with status code {response.status_code}")

def main():
    # List of URLs to process (example URLs here)
    with open("data/tmp/log_urls.txt", 'r') as f:
        urls = f.read().strip().split('\n')

    today = time.strftime("%Y-%m-%d")
    bucket_name = "ps-random-raw-data"

    # Create a list of tuples for each URL, where each tuple is (url, bucket_name, object_name)
    url_bucket_tuples = [(url, bucket_name, f"{today}/{url.split('/')[-1]}") for url in urls]
    for tup in url_bucket_tuples:
        print(tup)

    # Number of processes: can be set to the number of CPUs or adjusted as needed
    num_processes = 4

    # Create a pool of processes
    pool = Pool(processes=num_processes)

    # Map the upload function to the URLs
    pool.map(upload_url_contents_to_s3, url_bucket_tuples)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()