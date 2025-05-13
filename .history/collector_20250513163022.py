import os
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from an image using OCR
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# File processing function
def process_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        print("Processing PDF file...")
        return extract_text_from_pdf(file_path)

    elif file_extension == '.docx':
        print("Processing DOCX file...")
        return extract_text_from_docx(file_path)

    elif file_extension in ['.jpg', '.jpeg', '.png']:
        print("Processing image file...")
        return extract_text_from_image(file_path)

    else:
        raise ValueError("Unsupported file type!")

# Main function
if __name__ == "__main__":
    # Example file path
    file_path = input("Enter the file path: ")

    if os.path.exists(file_path):
        try:
            extracted_text = process_file(file_path)
            print("Extracted Text:\n")
            print(extracted_text)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("File does not exist. Please check the path.")
