import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
AWS_PROFILE = os.getenv("AWS_PROFILE")

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"