# wi25-ai-team-2

## Instructions

### 1. Run the Notebook in Google Colab

- Open the notebook OCR_RAG_Reranker_LLM.ipynb in Google Colab (https://colab.research.google.com/).
- Enable GPU:  
  Runtime > Change runtime type > Hardware accelerator > GPU
- Run all cells.

### 2. Setup ngrok in Your Backend

Use the following code snippet in your Colab or backend script to start ngrok with your auth token and expose your FastAPI/uvicorn server:

import nest_asyncio
import uvicorn
from pyngrok import ngrok

nest_asyncio.apply()

NGROK_AUTH_TOKEN = "<INSERT_NGROK_TOKEN>"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

ngrok.kill()  # Kill previous tunnels if any

public_url = ngrok.connect(8000)
print("Public URL:", public_url)

uvicorn.run("main:app", host="0.0.0.0", port=8000)

- Replace <INSERT_NGROK_TOKEN> with your actual ngrok auth token.

- The script will print a public URL like https://xxxx-xx-xx-xx-xx.ngrok.io — copy this URL.

### 3. Update Frontend App.js

- In the frontend folder, open App.js.

- Replace the existing API_BASE URL with your copied ngrok URL, for example:

const API_BASE = 'https://xxxx-xx-xx-xx-xx.ngrok.io';

### 4. Run Frontend

- In your terminal, navigate to the frontend folder.

- Run:

sudo npx expo start

- This will launch the React Native frontend, connected to your backend running through ngrok.

---

Notes:

- Make sure expo-cli is installed globally on your machine.
- When running in Colab, keep the session alive to maintain the ngrok tunnel.
- The React Native app uses the backend URL to upload files and query your OCR + RAG service.
