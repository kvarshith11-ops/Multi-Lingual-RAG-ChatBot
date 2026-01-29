import glob
from app.loaders.pdf_loader import PDFLoader

loader = PDFLoader()

# 🔎 Put Telugu text INSIDE quotes
needle = "అపకారికి ఉపకారం చేయరాదు"

for pdf in glob.glob("data/raw/*.pdf"):
    docs = loader.load(pdf)
    found = False
    for d in docs:
        if needle in d.content:
            print(
                "FOUND:",
                pdf,
                "| page:",
                d.metadata.get("page"),
                "| ocr:",
                d.metadata.get("ocr")
            )
            found = True
            break
    if not found:
        print("NOT FOUND:", pdf)
