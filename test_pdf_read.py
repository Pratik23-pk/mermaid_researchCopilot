from PyPDF2 import PdfReader

path = "/Users/pratikkanjilal/Desktop/Projects/research_copilot/pdfs/Pratik_Resume2024.pdf"  # <-- put exact filename here

reader = PdfReader(path)
print("Pages:", len(reader.pages))
for i, page in enumerate(reader.pages[:3]):
    text = page.extract_text() or ""
    print(f"\n--- Page {i+1} ---\n")
    print(text[:500])
