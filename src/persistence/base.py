"""
Module: src/persistence/base.py
Role: Abstract Repository Interfaces for Polyglot Storage.
Concepts: OOP (Abstraction, Dependency Inversion, Interface Segregation).
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from src.domain.models import CompanyProfile, ArticleRecord


class CompanyRepository(ABC):
    """
    Contract defining persistence operations for structured firmographics.
    Any concrete storage engine (MySQL, PostgreSQL, SQLite) must conform to this interface.
    """

    @abstractmethod
    def save(self, profile: CompanyProfile) -> None:
        """Persist or update (upsert) a company record."""
        pass

    @abstractmethod
    def get_by_id(self, company_id: str) -> Optional[CompanyProfile]:
        """Fetch a single company record by its primary key."""
        pass

    @abstractmethod
    def get_all(self) -> List[CompanyProfile]:
        """Retrieve all company records for dataset construction."""
        pass


class DocumentRepository(ABC):
    """
    Abstract contract for unstructured/semi-structured NoSQL document storage.
    """

    @abstractmethod
    def insert(self, record: ArticleRecord) -> None:
        """Persist or update a single text document."""
        pass

    @abstractmethod
    def get_by_company(self, company_id: str) -> List[ArticleRecord]:
        """Query all text documents associated with a specific startup."""
        pass