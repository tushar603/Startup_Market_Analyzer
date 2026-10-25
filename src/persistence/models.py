"""
Module: src/persistence/models.py
Role: MySQL Relational Schema for ML Feature Extraction and Company State.
"""
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from src.persistence.database import Base

class DBCompany(Base):
    __tablename__ = "companies"
    
    company_id = Column(String(100), primary_key=True, index=True)
    canonical_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_refresh_at = Column(DateTime, nullable=True)

    snapshots = relationship("DBCompanySnapshot", back_populates="company", cascade="all, delete")

class DBCompanySnapshot(Base):
    __tablename__ = "company_snapshots"
    
    snapshot_id = Column(String(64), primary_key=True) # MD5 Hash of company_id + date
    company_id = Column(String(100), ForeignKey("companies.company_id"), index=True)
    as_of_date = Column(DateTime, nullable=False, index=True)
    
    # Strictly structured features (Nulls preserved)
    total_funding_usd_m = Column(Float, nullable=True)
    employee_count = Column(Integer, nullable=True)
    company_age_years = Column(Float, nullable=True)
    industry_sector = Column(String(100), nullable=True)
    is_bootstrapped = Column(Boolean, nullable=True)
    
    # Link to MongoDB Document ID for full provenance details
    mongo_provenance_id = Column(String(100), nullable=True)
    
    company = relationship("DBCompany", back_populates="snapshots")