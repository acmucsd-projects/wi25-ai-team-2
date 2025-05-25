## wi25-ai-team-2

---

## Overview

This project implements a "Second Brain" app with:  
- Backend: FastAPI OCR + RAG pipeline prioritizing user docs over Wikipedia  
- Frontend: React Native (Expo) app for file upload and querying

---

## Setup Instructions

### 1. Backend Setup

- Create a `.env` file inside the `backend` folder with:

    NGROK_AUTH_TOKEN="your_ngrok_auth_token_here"

- Install backend dependencies:

    pip install -r requirements.txt

- Start the backend server by running:

    python run.py

> This will start the FastAPI backend and initialize the ngrok tunnel for external access.

---

### 2. Frontend Setup (React Native / Expo)

- Create a React Native frontend by running:

    sudo npx create-expo-app frontend --template blank

- Create a `.env` file in the `frontend` folder with your backend URL:

    LOCAL_BACKEND_URL="http://your_backend_url_here"

- Copy your existing `App.js` and `babel.config.js` into the new `frontend` folder.

- In your React Native code, import and use the backend URL from environment  
  variables using `react-native-dotenv`.

- Run the frontend with **web support**:

    cd frontend  
    npx expo start -w

> The `-w` flag opens the app in your browser.

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

- Keep the backend running with an active ngrok tunnel while using the frontend. 
- Install `expo-cli` globally if not already:

    npm install -g expo-cli

- Backend CORS is enabled for frontend requests.

---

## OCR + RAG Pipeline Summary

### OCR Phase
---------
- Users upload scanned documents (e.g., PDFs, images).
- Files are saved and OCR is applied to extract raw text (via `process_uploaded_files()`).
- Each extracted text chunk is **immediately summarized** using `facebook/bart-large-cnn` to reduce future token usage.
- Summarized text is appended line-by-line to a persistent file (`ocr_docs.txt`) for future queries.

### Query Phase
-----------
- User submits a natural language question via the `/query/` endpoint.
- The system loads summarized OCR content from `ocr_docs.txt`.

### Optional Padding with Wikipedia
-------------------------------
- If fewer than `top_k` user OCR docs are present, Wikipedia content is retrieved to supplement.
- Wikipedia docs are pre-embedded at startup using the `bge-base-en-v1.5` encoder and indexed with FAISS.
- These are retrieved by embedding similarity, then reranked using the `bge-reranker-large` cross-encoder.

### Summarization Step (Wikipedia Only)
-----------------------------------
- Retrieved Wikipedia docs are summarized individually using `facebook/bart-large-cnn`.
- Summarized Wikipedia docs are concatenated into `wiki_context`.

### LLM Answer Generation
---------------------
- A final prompt is constructed using the user's query, summarized Wikipedia context, and top summarized OCR chunks.
- This prompt is passed to the generator model (`deepcogito/cogito-v1-preview-llama-3B`).
- The generated answer is returned and saved to `answer.txt`.

### Code Logic Highlights
---------------------
- OCR summarization happens right after file upload in `process_uploaded_files()`.
- Query logic, Wikipedia fallback, summarization, and generation are in the `/query/` endpoint (`main.py`).
- You can configure `top_k`, thresholds, summarization model, or disable Wikipedia fallback.
