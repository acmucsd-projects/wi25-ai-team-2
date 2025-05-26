## WI25-AI-TEAM-2

This project implements a "Second Brain" system — a personal AI assistant that
answers questions based on your uploaded documents, with optional fallback
support from Wikipedia.

---

## Project Structure

### Backend (FastAPI + Ngrok)
- OCR + RAG (Retrieval-Augmented Generation) pipeline
- Prioritizes user-uploaded documents
- Fallback to Wikipedia only when needed

### Frontend (React Native + Expo)
- Mobile/web interface
- Supports file upload and natural language querying

---

## Setup Instructions

### 1. Backend Setup

1. Create a `.env` file in the `backend/` directory:

       NGROK_AUTH_TOKEN="your_ngrok_auth_token_here"

2. Install dependencies:

       pip install -r requirements.txt

3. Run the backend:

       python run.py

This starts the FastAPI server and exposes it using an ngrok tunnel.

---

### 2. Frontend Setup (React Native / Expo)

1. Create the frontend project:

       npx create-expo-app frontend --template blank

2. Create a `.env` file in `frontend/`:

       LOCAL_BACKEND_URL="http://your_backend_url_here"

3. Copy your existing `App.js` and `babel.config.js` into the `frontend/` folder.

4. Use `react-native-dotenv` to import environment variables.

5. Start the frontend in web mode:

       cd frontend
       npx expo start -w

To support this, install the Expo CLI if needed:

       npm install -g expo-cli

---

## Document Retrieval Logic

- Retrieves up to `top_k` user-uploaded documents based on similarity
- Filters out low-relevance user docs
- If fewer than `top_k` are relevant, supplements with Wikipedia documents
- Final answers prioritize user content; Wikipedia is used only as fallback

---

## OCR + RAG Pipeline

### OCR Phase
- Users upload scanned documents (PDFs/images)
- Text is extracted via OCR in `process_uploaded_files()`
- Each text chunk is summarized immediately with `facebook/bart-large-cnn`
- Summarized results are stored line-by-line in `ocr_docs.txt`

### Query Phase
- User submits a question to the `/query/` endpoint
- System loads summarized user docs from `ocr_docs.txt`
- Top relevant documents are selected using `bge-reranker-large`

### Wikipedia Fallback (Optional)
- If needed, Wikipedia docs are retrieved using FAISS index built on
  `bge-base-en-v1.5` embeddings
- Top docs are summarized using the same BART model

### Answer Generation
- A prompt is constructed from:
  - User question
  - Summarized OCR content
  - (Optional) summarized Wikipedia docs
- The generator model `deepcogito/cogito-v1-preview-llama-3B` produces the final answer
- Answer is saved to `answer.txt` and returned to the user

---

## Code Highlights

- `process_uploaded_files()` handles OCR and summarization
- `/query/` handles:
  - Retrieval of relevant documents
  - Reranking
  - Wikipedia fallback
  - Prompt construction and LLM generation
- Configurable parameters:
  - `top_k`
  - similarity thresholds
  - summarization model
  - Wikipedia fallback toggle

---

## Notes

- Keep the backend server and ngrok tunnel running during frontend use
- Make sure `.env` files are correctly configured in both frontend and backend
- CORS is enabled on the backend to support frontend requests
