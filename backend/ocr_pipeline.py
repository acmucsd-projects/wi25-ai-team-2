import os
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import re

# === Init OCR model ===
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# === PDF to Image ===
def pdf_to_images(pdf_path, dpi=300):
    return convert_from_path(pdf_path, dpi)

def save_images(pages, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, page in enumerate(pages):
        img_path = os.path.join(out_dir, f"page_{i}.jpg")
        page.save(img_path, 'JPEG')
        paths.append(img_path)
    return paths

# === OCR Text Extraction ===
def extract_text_with_paddleocr(img_path):
    result = ocr.ocr(img_path, cls=True)
    lines = [line[1][0] for line in result[0]]
    return "\n".join(lines)

# === Cleaning & Chunking ===
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def chunk_text(text, max_words=300):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def process_uploaded_files(file_paths, output_txt, output_pages, chunk_size=300):
    all_text = []

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        print(f"\n Processing: {filepath}")

        if filename.lower().endswith(".pdf"):
            pages = pdf_to_images(filepath)
            img_dir = os.path.join(output_pages, os.path.splitext(filename)[0])
            image_paths = save_images(pages, out_dir=img_dir)
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths = [filepath]
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        for img_path in image_paths:
            text = extract_text_with_paddleocr(img_path)
            all_text.append(clean_text(text))

    full_text = "\n\n".join(all_text)
    chunks = chunk_text(full_text, max_words=chunk_size)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))

    print(f"\n OCR complete. Output saved to: {output_txt}")
    return chunks
