import os
import re
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document
from odf.opendocument import load
from odf.text import P
import cv2
from pdf2image import convert_from_path
import numpy as np
from odf.text import Span
from odf import text, table
from odf.element import Element


# Set the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Add Ghostscript manually to the PATH
gs_path = r"C:\Program Files\gs\gs10.05.1\bin"
if gs_path not in os.environ["PATH"]:
    os.environ["PATH"] = gs_path + os.pathsep + os.environ["PATH"]

# Function to preprocess image for better OCR
def preprocess_image_for_ocr(image):
    # Convert to grayscale (OpenCV expects a numpy array)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Denoise the image (Optional but often helpful)
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    # Apply adaptive thresholding (better for poor quality images)
    adaptive_thresh = cv2.adaptiveThreshold(denoised_image, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)

    # Optionally, apply Gaussian Blur for better edge detection
    blurred_image = cv2.GaussianBlur(adaptive_thresh, (5, 5), 0)

    # Resize the image (optional)
    resized_image = cv2.resize(blurred_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    return resized_image

# Function to extract text from PDF with fallback to OCR
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()

            # If the page has text, use it
            if page_text:
                text += page_text
            else:
                # If no text is found, use OCR to extract text from images
                im = page.to_image()

                # Convert PIL image to numpy array for OpenCV processing
                open_cv_image = np.array(im.original)
                
                # Convert RGB to BGR (OpenCV expects BGR format)
                open_cv_image = open_cv_image[:, :, ::-1]
                
                # Preprocess image for better OCR results
                processed_image = preprocess_image_for_ocr(open_cv_image)

                # Use pytesseract to extract text from the image
                ocr_text = pytesseract.image_to_string(processed_image)
                text += ocr_text

    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_odt(odt_path):
    def get_text_recursive(element):
        text_content = ""
        for node in element.childNodes:
            if node.nodeType == node.TEXT_NODE:
                text_content += node.data
            elif isinstance(node, Element):
                text_content += get_text_recursive(node)
        return text_content

    doc = load(odt_path)
    all_text = []

    for elem in doc.getElementsByType(text.P) + doc.getElementsByType(text.H) + doc.getElementsByType(text.Span):
        para_text = get_text_recursive(elem).strip()
        if para_text:
            all_text.append(para_text)

    # Extract text from tables
    for table_elem in doc.getElementsByType(table.Table):
        for row in table_elem.getElementsByType(table.TableRow):
            row_text = []
            for cell in row.getElementsByType(table.TableCell):
                cell_paragraphs = cell.getElementsByType(text.P)
                for p in cell_paragraphs:
                    cell_text = get_text_recursive(p).strip()
                    if cell_text:
                        row_text.append(cell_text)
            if row_text:
                all_text.append(" | ".join(row_text))

    return "\n".join(all_text)


# Function to extract text from an image using OCR
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Function to process image files (direct OCR)
def process_image_file(image_path):
    img = cv2.imread(image_path)
    preprocessed_img = preprocess_image_for_ocr(img)
    text = pytesseract.image_to_string(preprocessed_img)
    return text

def is_low_quality_text(text: str, threshold_chars=200):
    """Determine if the extracted text is likely low quality."""
    # Check if text is empty or too short
    if not text.strip():
        return True
    if len(text.strip()) < threshold_chars:
        return True
    # Check for very low alphabetic content
    alpha_ratio = sum(c.isalpha() for c in text) / (len(text) + 1e-5)
    if alpha_ratio < 0.3:
        return True
    return False

# File processing function

def is_low_quality_text(text: str, threshold_chars=200):
    """Determine if the extracted text is likely low quality."""
    if not text.strip():
        return True
    if len(text.strip()) < threshold_chars:
        return True
    alpha_ratio = sum(c.isalpha() for c in text) / (len(text) + 1e-5)
    if alpha_ratio < 0.3:
        return True
    return False

def process_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        print("Processing PDF file...")
        text = extract_text_from_pdf(file_path)

    elif file_extension == '.docx':
        print("Processing DOCX file...")
        text = extract_text_from_docx(file_path)

    elif file_extension == '.odt':
        print("Processing ODT file...")
        text = extract_text_from_odt(file_path)

    elif file_extension in ['.jpg', '.jpeg', '.png']:
        print("Processing image file...")
        text = extract_text_from_image(file_path)

    else:
        raise ValueError("Unsupported file type!")

    # ✅ Check for poor quality after extraction
    if is_low_quality_text(text):
        print("⚠️ Warning: The data quality appears to be too low. Please upload a clearer version or better scan.")

    return text

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
