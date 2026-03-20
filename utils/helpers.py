from pypdf import PdfReader
import sys
import os

def extract_text_from_txt(file_obj) -> str:
    """Extract text from a .txt file object"""
    try:
        # Streamlit UploadedFile is essentially a BytesIO
        text = file_obj.getvalue().decode("utf-8")
        return text
    except Exception as e:
        return f"Error reading text file: {e}"

def extract_text_from_pdf(file_obj) -> str:
    """Extract text from a .pdf file object using pypdf"""
    try:
        reader = PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF file: {e}"

def process_uploaded_file(uploaded_file) -> str:
    """
    Takes a Streamlit UploadedFile and returns the extracted text.
    """
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            return extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.lower().endswith(".txt"):
            return extract_text_from_txt(uploaded_file)
        else:
            return "Unsupported file format. Please upload a .txt or .pdf file."
    except Exception as e:
        return f"Error processing file: {e}"
