## wi25-ai-team-2

### Instructions

1. Run the Notebook in Google Colab
-----------------------------------
- Open the notebook `OCR_RAG_Reranker_LLM.ipynb` in Google Colab: https://colab.research.google.com/
- Enable GPU acceleration:
  - Runtime > Change runtime type
  - Hardware accelerator > GPU
- Run all cells.

2. Set Up ngrok Authentication Token
------------------------------------
- In your backend directory, create a `.env` file.
- Add your ngrok auth token as follows:
  
  NGROK_AUTH_TOKEN=your_ngrok_auth_token_here

- Ensure your backend code loads this `.env` file and uses this environment variable.

3. Expose Backend via ngrok
---------------------------
- Run your backend server (FastAPI + uvicorn).
- The script will print a public URL like:
  
  https://xxxx-xx-xx-xx-xx.ngrok-free.app

- Copy this URL.

4. Set Up Frontend React Native App
-----------------------------------
- If you don’t have a React Native frontend yet, create one by running:

  sudo npx create-expo-app frontend --template blank

- Copy your existing `App.js` into the new `frontend` folder.
- Open `frontend/App.js` and replace the `API_BASE` URL with your ngrok URL, for example:

  const API_BASE = 'https://xxxx-xx-xx-xx-xx.ngrok-free.app';

5. Run Frontend
---------------
- Navigate to your frontend folder.
- Start the app with:

  sudo npx expo start

Notes:
------
- Make sure `expo-cli` is installed globally.
- Keep Colab sessions alive to maintain the ngrok tunnel if running backend there.
- The React Native app uses the backend URL to connect to your OCR + RAG service.