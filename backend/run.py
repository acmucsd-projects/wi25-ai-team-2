import os
import nest_asyncio
import uvicorn
from pyngrok import ngrok
from dotenv import load_dotenv
import main

# Load environment variables from .env file
load_dotenv()

# Apply patch for nested asyncio loops (required in notebooks or interactive environments)
nest_asyncio.apply()

# Set your ngrok authtoken (must be present in your .env file)
auth_token = os.getenv("NGROK_AUTH_TOKEN")
if not auth_token:
    raise ValueError("Missing NGROK_AUTH_TOKEN in environment variables or .env file.")
ngrok.set_auth_token(auth_token)

# Kill any existing ngrok tunnels to avoid conflict
ngrok.kill()

# Start a new ngrok tunnel on port 8000
public_url = ngrok.connect(8000, bind_tls=True).public_url
print(f"Public URL: {public_url}")

# Path to your frontend config file
frontend_config_path = '../frontend/config.js'

# Make sure the directory exists
os.makedirs(os.path.dirname(frontend_config_path), exist_ok=True)

# Write the dynamic API base URL to the frontend config.js
with open(frontend_config_path, 'w') as f:
    f.write(f"export const API_BASE = '{public_url}';\n")

print(f"Wrote backend URL to {frontend_config_path}")

# Set the public URL in your main app
main.public_url = public_url

# Run the FastAPI app with uvicorn
uvicorn.run(main.app, host="0.0.0.0", port=8000)
