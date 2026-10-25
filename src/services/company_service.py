"""
Module: src/services/company_service.py
Role: Polyglot persistence orchestrator mapping relational ML features to MySQL and raw media/provenance to MongoDB.
"""
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Optional
from sqlalchemy.orm import Session
from pymongo.database import Database

from src.persistence.models import DBCompany, DBCompanySnapshot
from src.ingestion.extractor import WebFirmographicExtractor, AmbiguousEntityError
from src.ingestion.news_client import LiveNewsIngestor
from src.domain.models import CompanyProfile, ArticleRecord

CACHE_EXPIRY = timedelta(days=7)

class PolyglotCompanyService:
    def __init__(self, mysql_db: Session, mongo_db: Database):
        self.mysql_db = mysql_db
        self.mongo_db = mongo_db
        self.extractor = WebFirmographicExtractor()
        self.ingestor = LiveNewsIngestor()

    def _generate_snapshot_id(self, company_id: str, as_of: datetime) -> str:
        payload = f"{company_id}_{as_of.strftime('%Y-%m-%d')}"
        return hashlib.md5(payload.encode()).hexdigest()

    def _save_unstructured_to_mongo(self, company_id: str, as_of_date: datetime, profile: CompanyProfile, articles: List[ArticleRecord]) -> str:
        """Stores raw articles and deep nested provenance JSONs to MongoDB."""
        snap_id = self._generate_snapshot_id(company_id, as_of_date)
        
        # 1. Save deep provenance document
        provenance_doc = {
            "snapshot_id": snap_id,
            "company_id": company_id,
            "as_of_date": as_of_date.isoformat(),
            "provenance_tree": profile.provenance,
            "raw_text_extracts": profile.provenance.get("raw_text_extracts", "")
        }
        prov_result = self.mongo_db.provenance.update_one(
            {"snapshot_id": snap_id},
            {"$set": provenance_doc},
            upsert=True
        )
        
        # 2. Save raw news articles
        if articles:
            article_docs = [
                {
                    "company_id": company_id,
                    "snapshot_id": snap_id,
                    "article_id": a.article_id,
                    "source": a.source,
                    "title": a.title,
                    "content": a.content,
                    "published_at": a.published_at.isoformat()
                } for a in articles
            ]
            # Clear old articles for this snapshot, insert new
            self.mongo_db.articles.delete_many({"snapshot_id": snap_id})
            self.mongo_db.articles.insert_many(article_docs)
            
        return str(snap_id)

    def get_or_refresh_data(self, raw_query: str, as_of_date: datetime) -> Tuple[CompanyProfile, List[ArticleRecord], bool, dict]:
        """
        Orchestrates polyglot retrieval. 
        Returns (CompanyProfile, Articles, is_fresh, provenance_dict).
        """
        # 1. Disambiguation
        canonical_name = self.extractor.validate_and_disambiguate(raw_query)
        company_id = canonical_name.strip().lower().replace(" ", "_").replace("-", "_")

        now = datetime.now(timezone.utc)
        db_company = self.mysql_db.query(DBCompany).filter(DBCompany.company_id == company_id).first()
        
        # 2. Cache Freshness Check via MySQL
        is_fresh = False
        if db_company and db_company.last_refresh_at:
            if (now - db_company.last_refresh_at) < CACHE_EXPIRY:
                is_fresh = True

        snap_id = self._generate_snapshot_id(company_id, as_of_date)

        if is_fresh:
            db_snap = self.mysql_db.query(DBCompanySnapshot).filter(DBCompanySnapshot.snapshot_id == snap_id).first()
            if db_snap:
                # Fetch Unstructured Data from MongoDB
                mongo_prov = self.mongo_db.provenance.find_one({"snapshot_id": snap_id})
                provenance_data = mongo_prov.get("provenance_tree", {}) if mongo_prov else {}
                
                mongo_articles = list(self.mongo_db.articles.find({"snapshot_id": snap_id}))
                articles = [
                    ArticleRecord(
                        article_id=doc["article_id"],
                        company_id=doc["company_id"],
                        source=doc["source"],
                        title=doc["title"],
                        content=doc["content"],
                        published_at=datetime.fromisoformat(doc["published_at"])
                    ) for doc in mongo_articles
                ]
                
                profile = CompanyProfile(
                    company_id=db_company.company_id,
                    company_name=db_company.canonical_name,
                    snapshot_date=db_snap.as_of_date.replace(tzinfo=timezone.utc),
                    total_funding_usd_m=db_snap.total_funding_usd_m,
                    employee_count=db_snap.employee_count,
                    company_age_years=db_snap.company_age_years,
                    industry_sector=db_snap.industry_sector,
                    is_bootstrapped=db_snap.is_bootstrapped,
                    provenance=provenance_data,
                    growth_rate=None 
                )
                return profile, articles, True, provenance_data

        # 3. Live Refresh (Cache Miss or Stale)
        profile = self.extractor.extract_profile(canonical_name, company_id, as_of_date)
        articles = self.ingestor.get_company_news(company_name=profile.company_name, company_id=company_id)
        
        # 4. Save Unstructured data to MongoDB
        mongo_prov_id = self._save_unstructured_to_mongo(company_id, as_of_date, profile, articles)
        
        # 5. Save Relational data to MySQL
        if not db_company:
            db_company = DBCompany(company_id=company_id, canonical_name=profile.company_name)
            self.mysql_db.add(db_company)
        
        db_company.last_refresh_at = now
        
        new_snap = DBCompanySnapshot(
            snapshot_id=snap_id,
            company_id=company_id,
            as_of_date=as_of_date.replace(tzinfo=None),
            total_funding_usd_m=profile.total_funding_usd_m,
            employee_count=profile.employee_count,
            company_age_years=profile.company_age_years,
            industry_sector=profile.industry_sector,
            is_bootstrapped=profile.is_bootstrapped,
            mongo_provenance_id=mongo_prov_id
        )
        self.mysql_db.merge(new_snap)
        self.mysql_db.commit()

        return profile, articles, False, profile.provenance