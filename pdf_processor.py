import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(file_path):
    """
    Loads a PDF, extracts text, and splits it into chunks.
    Returns empty list if PDF processing fails.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Check if any text was extracted at all
        if not documents or not any(doc.page_content.strip() for doc in documents):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("ERROR: No text was extracted from the PDF.")
            print("The PDF might be an image, a scan, or have an incompatible format.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return []  # Return an empty list to avoid further errors

        # Filter out empty pages
        valid_documents = [doc for doc in documents if doc.page_content.strip()]
        
        print("--- PDF PROCESSING DEBUG ---")
        print(f"Successfully loaded {len(valid_documents)} page(s) with content from the PDF.")
        if valid_documents:
            preview_text = valid_documents[0].page_content[:300].replace('\n', ' ').strip()
            print(f"First 300 characters preview: \n'{preview_text}...'")
        print("----------------------------")

        # Use improved text splitter settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased chunk size for better context
            chunk_overlap=300,  # Increased overlap for better continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separation hierarchy
        )
        
        texts = text_splitter.split_documents(valid_documents)
        
        print(f"Created {len(texts)} text chunks for processing.")
        
        return texts
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return []