import os
import json

from mwxml import Dump, Page
import mwparserfromhell

PROJECT_PATH = os.path.join(os.path.dirname(__file__), '../')
DATA = os.path.join(PROJECT_PATH, "data")
RAW_FILES = os.path.join(DATA, "raw")

index_file = os.path.join(RAW_FILES, "enwiki-20250123-pages-articles-multistream-index2.txt")
xml_file = os.path.join(RAW_FILES, "enwiki-20250123-pages-articles-multistream2.xml")

def process_dump(xml_path, year, ind):
	dump = Dump.from_file(open(xml_path, 'rb'))

	output_path = os.path.join(DATA, f"documents{year}-{ind}.json")
	with open(output_path, 'w', encoding='utf-8') as file:
		for page in dump.pages:
			if page.redirect:
				continue
			for revision in page:
				wikitext = revision.text
				text = mwparserfromhell.parse(wikitext).strip_code()

				entry = {
					"id": page.id,
					"title": page.title,
					"text": text,
				}
				file.write(json.dumps(entry, ensure_ascii=False) + '\n')
				break  # Just show the latest revision

process_dump(xml_file, 2025, 2)
