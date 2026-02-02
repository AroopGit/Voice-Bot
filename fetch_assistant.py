import requests
import json
import os

url = "https://api.vapi.ai/assistant/575006b6-5799-4383-98bd-2d01c5a93f14"
headers = {"Authorization": "Bearer cce95178-4d06-4e2d-a583-344e91c741e2"}

try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open("d:/Voice_bot/assistant_config.json", "w", encoding="utf-8") as f:
            json.dump(response.json(), f, indent=2)
        print("Success")
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
