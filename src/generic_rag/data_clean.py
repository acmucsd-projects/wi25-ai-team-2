import json
import re
import os

PROJECT_PATH = os.path.join(os.path.dirname(__file__), '../..')
RAW_FILES = os.path.join(PROJECT_PATH, "data", "raw")
CLEAN_FILES = os.path.join(PROJECT_PATH, "data", "clean")


def isValidPath(path):
    invalid_chars = r'[<>:"/\\|?*]'
    return not bool(re.search(invalid_chars, os.path.basename(path)))

def scanPage(page):
	titleStartMatch = re.search(r"<title>", page)
	titleEndMatch = re.search(r"</title>", page)
	title = page[titleStartMatch.end() : titleEndMatch.start()]

	print("Title found: " + title)

	textStartMatch = re.search(r"<text\b[^>]*?>", page)
	textEndMatch = re.search(r"</text>", page)
	text = page[textStartMatch.end() : textEndMatch.start()]
	
	writePath = os.path.join(CLEAN_FILES, title.replace("/","").replace("\\","") + ".txt")
	if (not isValidPath(writePath)):
		return
	with open(writePath, "w", encoding="utf-8") as f:
		print("Writing to " + writePath)
		f.write(text)

def scanFile(filename):
	print("Reading file: " + filename)
	try:
		with open(filename, "r", encoding="utf-8") as file:
			content = file.read()

			pageStartTags = [match.start() for match in re.finditer(re.escape("<page>"), content)]
			pageEndTags = [match.end() for match in re.finditer(re.escape("</page>"), content)]
			pages = [content[pageStartTags[i] : pageEndTags[i]] for i in range(len(pageStartTags))]
			
			print("Found " + str(len(pages)) + " pages.")
			for page in pages:
				scanPage(page)

	except FileNotFoundError:
		print("File not found.")
	except Exception as e:
		print(f"An error occurred: {e}")



def main():
	rawFiles = [
		"enwiki-20250123-pages-articles-multistream2.xml",
		"enwiki-20250401-pages-articles-multistream7.xml",
	]
	for fileName in rawFiles:
		scanFile(os.path.join(RAW_FILES, fileName))

if __name__ == "__main__":
	main()