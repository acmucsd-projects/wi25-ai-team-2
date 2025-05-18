import nest_asyncio
import uvicorn
from pyngrok import ngrok
import os
from dotenv import load_dotenv

nest_asyncio.apply()

# Load environment variables from .env
load_dotenv()

NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
if not NGROK_AUTH_TOKEN:
    raise ValueError("NGROK_AUTH_TOKEN not set in .env")

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Restart ngrok tunnel
ngrok.kill()
public_url = ngrok.connect(8000).public_url
print("Public URL:", public_url)

# Write API base to frontend/config.js
frontend_config_path = os.path.join(os.path.dirname(__file__), "../frontend/config.js")
with open(frontend_config_path, "w") as f:
    f.write(f'export const API_BASE = "{public_url}";\n')

# Run FastAPI app
uvicorn.run("main:app", host="0.0.0.0", port=8000)
