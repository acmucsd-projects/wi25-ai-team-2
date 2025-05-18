## wi25-ai-team-2

---

## Overview

This project implements a "Second Brain" app with:  
- Backend: FastAPI OCR + RAG pipeline prioritizing user docs over Wikipedia  
- Frontend: React Native (Expo) app for file upload and querying

---

## Setup Instructions

### 1. Backend Setup

- Create a `.env` file inside the backend folder with:

    NGROK_AUTH_TOKEN="your_ngrok_auth_token_here"

- Install backend dependencies:

    pip install -r requirements.txt

- Start the backend server (e.g., uvicorn):

    uvicorn main:app --host 0.0.0.0 --port 8000

---

### 2. Frontend Setup (React Native / Expo)

- Create a `.env` file in the frontend folder with your backend URL:

    LOCAL_BACKEND_URL="http://your_backend_url_here"

- If you don’t have a React Native frontend yet, create one by running:

    sudo npx create-expo-app frontend --template blank

- Copy your existing `App.js` and `babel.config.js` into the new `frontend` folder.

- In your React Native code, import and use your backend URL from environment  
  variables using your preferred method (e.g., `react-native-dotenv`).

- Run the frontend:

    cd frontend  
    npx expo start

- Use Expo Go on your device or emulator to run the app.

---

## Document Retrieval Logic

- Backend prioritizes user-uploaded documents when answering queries.  
- It tries to retrieve up to `top_k` user docs.  
- Filters out low-relevance user docs.  
- If fewer than `top_k` relevant user docs remain, supplements with Wikipedia docs  
  to reach `top_k` total.  
- Ensures answers mainly rely on user content; Wikipedia is fallback only.

---

## Notes

- Keep backend running with active ngrok tunnel while using frontend.  
- Update frontend `.env` if ngrok URL changes.  
- Install `expo-cli` globally if not already:

    npm install -g expo-cli

- Backend CORS is enabled for frontend requests.