import json
from pathlib import Path
from typing import Any, List, Dict

from app.config import (
    RAW_DOCS_DIR, CLEAR_DOCS_DIR, CHUNK_OVERLAP_RATIO
)
from docling.document_converter import DocumentConverter


class DocumentChunkingService:
    def __init__(self, raw_docs_dir: str = None, clear_docs_dir: str = None):
        self.raw_docs_dir = Path(raw_docs_dir or RAW_DOCS_DIR)
        self.clear_docs_dir = Path(clear_docs_dir or CLEAR_DOCS_DIR)
        self.converter = DocumentConverter()

        self.raw_docs_dir.mkdir(exist_ok=True)
        self.clear_docs_dir.mkdir(exist_ok=True)

    def _recursively_split_text(self, text: str, max_size: int, overlap: int) -> List[str]:
        """
        Recursively splits text into chunks if it exceeds max_size.s
        """
        chunks = []

        if len(text) <= max_size:
            chunks.append(text)
            return chunks

        split_point = self._find_best_split_point(text, max_size)

        if split_point == -1:
            split_point = max_size

        first_chunk = text[:split_point].strip()
        if first_chunk:
            chunks.append(first_chunk)

        remaining_text = text[split_point - overlap:] if split_point > overlap else text[split_point:]
        if remaining_text:
            chunks.extend(self._recursively_split_text(remaining_text, max_size, overlap))

        return chunks

    def _find_best_split_point(self, text: str, max_size: int) -> int:
        """
        Finds optimal point for text splitting.
        Prefers sentence endings, then commas, then word spaces.
        """
        sentence_endings = ['.', '!', '?', '。', '！', '？']
        for i in range(min(max_size, len(text)) - 1, max(0, max_size - 100), -1):
            if text[i] in sentence_endings and (i + 1 >= len(text) or text[i + 1] in [' ', '\n', '"', "'"]):
                return i + 1

        for i in range(min(max_size, len(text)) - 1, max(0, max_size - 50), -1):
            if text[i] == ',' and text[i + 1] == ' ':
                return i + 1

        for i in range(min(max_size, len(text)) - 1, max(0, max_size - 30), -1):
            if text[i] == ' ':
                return i + 1

        return -1

    def _extract_text_elements(self, items: Any) -> List[Dict[str, Any]]:
        """
        Extracts text from Docling document items.
        """
        paragraphs = []

        for item in items:
            text_content = item.text if hasattr(item, 'text') else str(item)

            if text_content and text_content.strip():
                element_type = type(item).__name__

                paragraphs.append({
                    'text': text_content.strip(),
                    'title': f"{element_type}_{len(paragraphs) + 1}",
                    'element_type': element_type,
                    'order': len(paragraphs)
                })

        return paragraphs

    def chunk_document_from_file(self, filename: str, max_chunk_size: int) -> List[Dict[str, Any]]:
        """
        Main document processing method.
        """
        file_path = self.raw_docs_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        print(f"Processing file: {filename}")

        result = self.converter.convert(str(file_path))

        chunks = []
        chunk_id = 0
        overlap_size = int(max_chunk_size * CHUNK_OVERLAP_RATIO) 

        if hasattr(result.document, 'texts'):
            paragraphs = self._extract_text_elements(result.document.texts)
        else:
            full_text = str(result.document)
            paragraphs = [{
                'text': full_text,
                'title': 'Full_Document',
                'element_type': 'Document',
                'order': 0
            }]

        print(f"Found {len(paragraphs)} text elements for processing")

        for para_data in paragraphs:
            paragraph = para_data['text']
            section_title = para_data['title']
            element_type = para_data['element_type']

            if not paragraph.strip():
                continue

            if len(paragraph) <= max_chunk_size:
                chunk_data = {
                    "chunk_id": chunk_id,
                    "section_title": section_title,
                    "element_type": element_type,
                    "text": paragraph,
                    "chunk_size": len(paragraph),
                    "is_split": False,
                    "original_paragraph_size": len(paragraph)
                }
                chunks.append(chunk_data)
                chunk_id += 1
            else:
                paragraph_chunks = self._recursively_split_text(
                    paragraph, max_chunk_size, overlap_size
                )

                for i, chunk_text in enumerate(paragraph_chunks):
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "section_title": section_title,
                        "element_type": element_type,
                        "text": chunk_text,
                        "chunk_size": len(chunk_text),
                        "is_split": True,
                        "split_part": f"{i+1}/{len(paragraph_chunks)}",
                        "original_paragraph_size": len(paragraph)
                    }
                    chunks.append(chunk_data)
                    chunk_id += 1

        output_filename = f"{Path(filename).stem}_chunks.json"
        output_path = self.clear_docs_dir / output_filename

        output_data = {
            "source_file": filename,
            "max_chunk_size": max_chunk_size,
            "overlap_size": overlap_size,
            "total_chunks": len(chunks),
            "chunks": chunks
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Processing completed. Created {len(chunks)} chunks.")
        print(f"Result saved to: {output_path}")

        return chunks

    def calculate_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns statistics for created chunks.
        """
        if not chunks:
            return {}

        total_size = sum(chunk['chunk_size'] for chunk in chunks)
        split_chunks = [chunk for chunk in chunks if chunk.get('is_split', False)]

        return {
            "total_chunks": len(chunks),
            "split_chunks": len(split_chunks),
            "avg_chunk_size": total_size / len(chunks),
            "min_chunk_size": min(chunk['chunk_size'] for chunk in chunks),
            "max_chunk_size": max(chunk['chunk_size'] for chunk in chunks),
            "split_percentage": (len(split_chunks) / len(chunks)) * 100 if chunks else 0
        }