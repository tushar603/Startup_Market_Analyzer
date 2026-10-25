"""
Module: base.py
Role: Abstract Interface for Ingestion & OS Process Resource Management.
Concepts: OOP (Abstraction, Strategy Pattern), OS (Child Process Lifecycle, Context Managers).
"""

from abc import ABC, abstractmethod
from typing import Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from src.domain.models import CompanyProfile


class BaseStructuredExtractor(ABC):
    """
    Abstract Strategy: Every structured source (Growjo, Crunchbase, etc.)
    must conform to this contract.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._driver: Optional[webdriver.Chrome] = None

    def __enter__(self):
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        self._driver = webdriver.Chrome(options=options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._driver:
            self._driver.quit()
            self._driver = None

    @abstractmethod
    def extract(self, identifier: str) -> CompanyProfile:
        pass