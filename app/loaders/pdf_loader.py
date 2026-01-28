from typing import List
from pypdf import PdfReader
from app.loaders.base_loader import BaseLoader
from app.models import Document

class PDFLoader(BaseLoader):

    def load(self, file_path: str) -> List[Document]:
        reader = PdfReader(file_path)
        documents = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1
                        }
                    )
                )

        return documents
