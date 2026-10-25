"""
Text Chunking Module (RAG-7)
Divides large documents into smaller overlapping text chunks.
Prevents context window overflow and information loss near boundaries.
"""

from typing import List, Dict, Any


def create_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Splits text into chunks of given size with specified overlap.
    Follows Step 13 & 30 specifications with parameter guards.
    """
    if not text or not text.strip():
        return []

    # Safeguard parameters
    if chunk_size <= 0:
        chunk_size = 1500
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    clean_text = text.strip()
    text_length = len(clean_text)

    if text_length <= chunk_size:
        return [clean_text]

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start += step

    return chunks


def chunk_document_pages(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 1500,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Takes list of page dictionaries from document_loader and splits them into
    traceable chunks with metadata: source filename, page, chunk_id, and text.
    """
    all_chunks = []
    chunk_counter = 0

    for page_info in pages_data:
        text = page_info.get("text", "").strip()
        source = page_info.get("source", "unknown")
        page_num = page_info.get("page", 1)

        if not text:
            continue

        raw_chunks = create_chunks(text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_counter += 1
            chunk_id = f"{source}_p{page_num}_c{idx+1}"
            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source": source,
                "page": page_num,
                "chunk_index": idx + 1,
                "char_length": len(chunk_text)
            })

    return all_chunks
