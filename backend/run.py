import os
import nest_asyncio
import uvicorn
from pyngrok import ngrok
import main  # your FastAPI app

# Apply the patch to allow nested event loops
nest_asyncio.apply()

# Kill existing ngrok tunnels (optional but avoids errors)
ngrok.kill()

# Connect ngrok to port 8000
public_url = ngrok.connect(8000, bind_tls=True).public_url
print(f"Public URL: {public_url}")
main.public_url = public_url

# Run the app with uvicorn
uvicorn.run(main.app, host="0.0.0.0", port=8000)
