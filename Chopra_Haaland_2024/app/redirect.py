# This code creates a redirects the published link for every interviewee to a unique URL/interview to avoid that the interview has already been run.
# Note that to use this script it is necessary to create another S3 bucket "counter.txt" that just counts the times the link has been redirected. 
# "counter.txt" can be started with the entry "0". Then, with every additional click on the link the count increases by 1.
# This is used to create the unique links. It might be that you have to adjust the policies in AWS slightly.

import boto3
import json

# Configuration
BUCKET_NAME = 'qualtrics-user-counter'
OBJECT_KEY = 'counter.txt'
BASE_URL = "REPLACE_WITH_QUALTRICS_URL"
INTERVIEW_ID = "STOCK_MARKET"
INTERVIEW_ENDPOINT = "https://<SOME_AWS_ID>.execute-api.<AWS_REGION>.amazonaws.com/Prod/"

s3 = boto3.client('s3')

def get_next_user_id():
    # Read current count
    counter_file = s3.get_object(Bucket=BUCKET_NAME, Key=OBJECT_KEY)
    counter_str = counter_file['Body'].read().decode('utf-8').strip()
    current_count = int(counter_str)

    # Increment count
    next_count = current_count + 1

    # Save back to S3
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=OBJECT_KEY,
        Body=str(next_count)
    )

    return f"user_SM_{next_count}"

def lambda_handler(event, context):
    try:
        user_id = get_next_user_id()

        redirect_url = (
            f"{BASE_URL}"
            f"?user_id={user_id}"
            f"&interview_id={INTERVIEW_ID}"
            f"&interview_endpoint={INTERVIEW_ENDPOINT}"
        )

        return {
            "statusCode": 302,
            "headers": {
                "Location": redirect_url
            },
            "body": "",
            "isBase64Encoded": False
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }