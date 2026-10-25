"""
Module: src/persistence/nosql_repo.py
Role: Document Store Implementation for Unstructured Media Feeds.
Concepts: 
  - DBMS: Document Store Pattern, Partitioning by Company ID.
  - OS: Buffered Disk I/O, Path Verification.
  - OOP: Concrete implementation of DocumentRepository.
"""

import os
import json
from typing import List
from src.domain.models import ArticleRecord
from src.persistence.base import DocumentRepository


class JSONDocumentRepository(DocumentRepository):
    def __init__(self, storage_dir: str = "data/documents"):
        self.storage_dir = storage_dir
        self._ensure_storage_ready()

    def _ensure_storage_ready(self) -> None:
        """Guarantees the storage directory exists on the OS filesystem."""
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_partition_path(self, company_id: str) -> str:
        """Returns the file partition path for a given company."""
        return os.path.join(self.storage_dir, f"{company_id.strip().lower()}_docs.json")

    def insert(self, record: ArticleRecord) -> None:
        """
        Inserts or updates an unstructured record inside the company's document partition.
        """
        file_path = self._get_partition_path(record.company_id)
        documents = []

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    documents = json.load(f)
            except json.JSONDecodeError:
                documents = []

        # Upsert document by article_id
        doc_dict = record.to_dict()
        existing_idx = next(
            (idx for idx, item in enumerate(documents) if item.get("article_id") == record.article_id),
            None
        )

        if existing_idx is not None:
            documents[existing_idx] = doc_dict
        else:
            documents.append(doc_dict)

        # OS Buffered write
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

    def get_by_company(self, company_id: str) -> List[ArticleRecord]:
        file_path = self._get_partition_path(company_id)
        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        return [
            ArticleRecord(
                article_id=doc["article_id"],
                company_id=doc["company_id"],
                source=doc["source"],
                title=doc["title"],
                content=doc["content"],
                published_at=doc["published_at"],
                metadata=doc.get("metadata", {})
            )
            for doc in raw_docs
        ]