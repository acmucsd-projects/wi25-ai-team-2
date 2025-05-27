import os
import nest_asyncio
import uvicorn
from pyngrok import ngrok
from dotenv import load_dotenv
import main
import requests

load_dotenv()
nest_asyncio.apply()

auth_token = os.getenv("NGROK_AUTH_TOKEN")
if not auth_token:
    raise ValueError("Missing NGROK_AUTH_TOKEN in environment variables or .env file.")
ngrok.set_auth_token(auth_token)
ngrok.kill()

public_url = ngrok.connect(8000, bind_tls=True).public_url
print(f"Public URL: {public_url}")

main.public_url = public_url

# Write new ngrok URL to file for your app to read
with open("ngrok_url.txt", "w") as f:
    f.write(public_url)

JSONBIN_URL = os.getenv("JSONBIN_URL")
JSONBIN_API_KEY = os.getenv("JSONBIN_API_KEY")

def update_jsonbin_url(new_url: str):
    headers = {
        "Content-Type": "application/json",
        "X-Master-Key": JSONBIN_API_KEY,
    }
    data = {
        "url": new_url
    }
    response = requests.put(JSONBIN_URL, json=data, headers=headers)
    if response.status_code == 200:
        print("JSONBin updated successfully!")
    else:
        print(f"Failed to update JSONBin: {response.status_code} - {response.text}")

if JSONBIN_API_KEY and JSONBIN_URL:
    update_jsonbin_url(public_url)
else:
    print("JSONBIN_API_KEY or JSONBIN_URL not set, skipping JSONBin update")

uvicorn.run(main.app, host="0.0.0.0", port=8000)
