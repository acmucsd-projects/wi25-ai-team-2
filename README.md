README for wi25-ai-team-2

Instructions:

1. Run the Notebook in Google Colab
-----------------------------------
- Open the notebook OCR_RAG_Reranker_LLM.ipynb in Google Colab: https://colab.research.google.com/
- Enable GPU:
  Runtime > Change runtime type > Hardware accelerator > GPU
- Run all cells.

2. Setup ngrok Authentication Token
-----------------------------------
- Create a `.env` file in your backend directory.
- Add your ngrok auth token in the `.env` file as follows:

  NGROK_AUTH_TOKEN=your_ngrok_auth_token_here

- Make sure your backend code loads the `.env` file and uses this environment variable for ngrok.

3. Expose Backend via ngrok
---------------------------
- Run your backend (FastAPI + uvicorn).
- The script will print a public URL like:

  https://xxxx-xx-xx-xx-xx.ngrok-free.app

- Copy this URL.

4. Setup Frontend React Native App
----------------------------------
- If you don’t have a React Native frontend yet, create one from scratch:

  sudo npx create-expo-app frontend --template blank

- Copy your existing `App.js` file into the newly created `frontend` folder.

- Open `frontend/App.js` and replace the existing `API_BASE` URL with the ngrok public URL you copied, for example:

  const API_BASE = 'https://xxxx-xx-xx-xx-xx.ngrok-free.app';

5. Run Frontend
---------------
- Create or open your React Native frontend folder.
- Run the frontend with:

  sudo npx expo start

Notes:
------
- Ensure `expo-cli` is installed globally.
- Keep Colab sessions alive to maintain the ngrok tunnel if running there.
- The React Native app uses the backend URL to connect and interact with your OCR + RAG service.
