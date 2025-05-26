import os
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from models import summarize 
from utils import chunk_text
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

def process_uploaded_files(
	file_paths,
	output_txt=None,
	output_pages=None,
	summarizer_tokenizer=None,
	summarizer_model=None
):
	all_docs = []

	if output_txt is not None:
		os.makedirs(os.path.dirname(output_txt), exist_ok=True)

	for filepath in file_paths:
		filename = os.path.basename(filepath)
		print(f"\n Processing: {filepath}")

		if filename.lower().endswith(".pdf"):
			pages = pdf_to_images(filepath)
			img_dir = os.path.join(output_pages or "output_pages", os.path.splitext(filename)[0])
			image_paths = save_images(pages, out_dir=img_dir)
		elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
			image_paths = [filepath]
		else:
			print(f"Skipping unsupported file: {filename}")
			continue

		all_text = ""
		for img_path in image_paths:
			all_text += clean_text(extract_text_with_paddleocr(img_path))

		lines = chunk_text(all_text)

		# Always summarize
		summarized_lines = []
		for i, line in enumerate(lines):
			print(f"Summarizing chunk {i+1}/{len(lines)}...")
			summarized_line = summarize(line, summarizer_tokenizer, summarizer_model)
			summarized_lines.append(summarized_line)
		lines = summarized_lines

		all_docs.extend(lines)

		if output_txt:
			with open(output_txt, "a", encoding="utf-8") as f:
				for line in lines:
					f.write(line + "\n")

	if output_txt:
		print(f"\n OCR complete. Output saved to: {output_txt}")

	return all_docs