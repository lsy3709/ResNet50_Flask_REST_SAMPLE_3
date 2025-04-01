from dotenv import load_dotenv
load_dotenv()
import os  # ✅ 반드시 있어야 함

AWS_BUCKET = os.getenv("AWS_STORAGE_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_S3_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_S3_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_S3_REGION_NAME")
