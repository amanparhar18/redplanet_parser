import os
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document
from odf.opendocument import load
from odf.text import P
import cv2
from pdf2image import convert_from_path

# Set the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Add Ghostscript manually to the PATH
gs_path = r"C:\Program Files\gs\gs10.05.1\bin"
if gs_path not in os.environ["PATH"]:
    os.environ["PATH"] = gs_path + os.pathsep + os.environ["PATH"]

# Function to preprocess image for better OCR
def preprocess_image_for_ocr(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use thresholding to make the image more binary (black and white)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    
    # Resize the image to improve OCR recognition
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    return resized

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

# Function to extract text from ODT
def extract_text_from_odt(odt_path):
    doc = load(odt_path)
    text = ""
    for paragraph in doc.getElementsByType(P):
        text += paragraph.firstChild.data + "\n"
    return text

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

# File processing function
def process_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        print("Processing PDF file...")
        return extract_text_from_pdf(file_path)

    elif file_extension == '.docx':
        print("Processing DOCX file...")
        return extract_text_from_docx(file_path)

    elif file_extension == '.odt':
        print("Processing ODT file...")
        return extract_text_from_odt(file_path)

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
