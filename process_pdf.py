import glob
import uuid
import pandas as pd
import tiktoken
import fitz  # PyMuPDF
import re
import os
from langchain_openai import AzureChatOpenAI
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm 

def extract_clean_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        raw_text = page.get_text()
        raw_text = re.sub(r"^\s*strana \d+", "", raw_text, flags=re.MULTILINE)  # Remove page numbers
        raw_text = re.sub(r"^\d+\s+Z\u00c1KON.*", "", raw_text, flags=re.MULTILINE)  # Remove law number headers
        text += raw_text + "\n"
    cleaned_text = re.sub(r"\n{2,}", "\n", text)
    cleaned_text = re.sub(r" +", " ", cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def get_header(
    text: str,
    summarizer_instructions: str = "Return what law is this and nothing else",
    chunk_tokens: int = 4096
) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be a non-empty string.")
    enc = tiktoken.get_encoding("o200k_base")
    ids = enc.encode(text)
    if not ids:
        raise ValueError("Cannot encode text.")
    chunk_ids = ids[:chunk_tokens]
    chunk = enc.decode(chunk_ids)
    prompt = f"{summarizer_instructions}:\n\n{chunk}"
    try:
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini",  
            api_version="2024-12-01-preview",
            temperature=0,
            max_tokens=200,
            timeout=30,
            max_retries=2,
        )
        summary = llm.invoke(prompt).content.strip()
    except Exception as e:
        print(f"LLM summary failed: {e}")
        summary = ""
    return summary

def split_document(raw_texts, file_name="", header=""):
    documents = [Document(page_content=raw_texts)]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    chunk_docs = text_splitter.split_documents(documents)
    for idx, doc in enumerate(chunk_docs):
        doc.metadata['name'] = file_name
        doc.metadata['header'] = header
        doc.metadata['chunk_id'] = uuid.uuid4().hex
    return chunk_docs

def convert_chunks_to_df(chunks, header=""):
    rows = []
    for chunk in chunks:
        row = {
            "text": header + chunk.page_content,
            **chunk.metadata,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def main():
    dir_paths = glob.glob("e-sbirka_data/e-sbirka_data/*.pdf")
    all_chunks = []
    for pdf_path in tqdm(dir_paths, desc="Processing PDFs"):
        file_name = os.path.basename(pdf_path)
        print(f"Processing: {file_name}")
        clean_text = extract_clean_text_from_pdf(pdf_path)
        header = get_header(clean_text)
        chunks = split_document(clean_text, file_name=file_name, header=header)
        all_chunks.extend(chunks)
        print(f"  Added {len(chunks)} chunks.")
    
    final_df = convert_chunks_to_df(all_chunks, header)
    final_df.to_csv("all_chunks.csv", index=False)
    # Hoặc lưu .parquet nếu muốn:
    # final_df.to_parquet("all_chunks.parquet", index=False)
    print(f"Đã lưu tất cả chunk vào all_chunks.csv ({len(final_df)} dòng)")

if __name__ == "__main__":
    main()
