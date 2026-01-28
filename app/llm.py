import json
import boto3
from app.config import AWS_REGION, AWS_PROFILE

CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

class ClaudeClient:
    def __init__(self):
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        self.client = session.client("bedrock-runtime")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens
        }

        response = self.client.invoke_model(
            modelId=CLAUDE_MODEL_ID,
            body=json.dumps(payload).encode("utf-8"),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
