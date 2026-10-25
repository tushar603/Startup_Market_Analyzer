"""
Module: src/persistence/mysql_repo.py
Role: Concrete MySQL Persistence Implementation with Automatic Database Provisioning.
"""

from typing import Optional, List
import mysql.connector
from mysql.connector import pooling
from src.domain.models import CompanyProfile
from src.persistence.base import CompanyRepository


class MySQLCompanyRepository(CompanyRepository):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "your_mysql_password",
        database: str = "startup_market_analyzer",
        pool_name: str = "sma_mysql_pool",
        pool_size: int = 5
    ):
        self.db_name = database
        self.db_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
        }

        # 1. Self-healing DDL: Ensure database exists before creating the pool
        self._ensure_database_exists()

        # 2. Add database to config and initialize pool
        self.db_config["database"] = self.db_name
        self._pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name=pool_name,
            pool_size=pool_size,
            **self.db_config
        )

        # 3. Initialize table schema
        self._init_db()

    def _ensure_database_exists(self) -> None:
        """Connects to MySQL server instance to guarantee DB catalog existence."""
        raw_conn = mysql.connector.connect(**self.db_config)
        try:
            with raw_conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name};")
            raw_conn.commit()
        finally:
            raw_conn.close()

    def _get_connection(self):
        return self._pool.get_connection()

    def _init_db(self) -> None:
        ddl_query = """
        CREATE TABLE IF NOT EXISTS companies (
            company_id VARCHAR(100) PRIMARY KEY,
            company_name VARCHAR(255) NOT NULL,
            total_funding_usd_m DOUBLE,
            employee_count INT,
            growth_rate DOUBLE,
            company_age_years INT,
            industry_sector VARCHAR(100),
            headquarters VARCHAR(255),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(ddl_query)
            conn.commit()
        finally:
            conn.close()

    def save(self, profile: CompanyProfile) -> None:
        upsert_query = """
        INSERT INTO companies (
            company_id, company_name, total_funding_usd_m, employee_count,
            growth_rate, company_age_years, industry_sector, headquarters
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            company_name = VALUES(company_name),
            total_funding_usd_m = VALUES(total_funding_usd_m),
            employee_count = VALUES(employee_count),
            growth_rate = VALUES(growth_rate),
            company_age_years = VALUES(company_age_years),
            industry_sector = VALUES(industry_sector),
            headquarters = VALUES(headquarters);
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    upsert_query,
                    (
                        profile.company_id,
                        profile.company_name,
                        profile.total_funding_usd_m,
                        profile.employee_count,
                        profile.growth_rate,
                        profile.company_age_years,
                        profile.industry_sector,
                        profile.headquarters,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def get_by_id(self, company_id: str) -> Optional[CompanyProfile]:
        query = "SELECT * FROM companies WHERE company_id = %s;"
        conn = self._get_connection()
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(query, (company_id.strip().lower(),))
                row = cursor.fetchone()
                if not row:
                    return None
                return CompanyProfile(
                    company_id=row["company_id"],
                    company_name=row["company_name"],
                    total_funding_usd_m=row["total_funding_usd_m"],
                    employee_count=row["employee_count"],
                    growth_rate=row["growth_rate"],
                    company_age_years=row["company_age_years"],
                    industry_sector=row["industry_sector"],
                    headquarters=row["headquarters"],
                )
        finally:
            conn.close()

    def get_all(self) -> List[CompanyProfile]:
        query = "SELECT * FROM companies;"
        conn = self._get_connection()
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                return [
                    CompanyProfile(
                        company_id=row["company_id"],
                        company_name=row["company_name"],
                        total_funding_usd_m=row["total_funding_usd_m"],
                        employee_count=row["employee_count"],
                        growth_rate=row["growth_rate"],
                        company_age_years=row["company_age_years"],
                        industry_sector=row["industry_sector"],
                        headquarters=row["headquarters"],
                    )
                    for row in rows
                ]
        finally:
            conn.close()