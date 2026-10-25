"""
Module: src/persistence/database.py
Role: Dual-database connection manager (MySQL for Relational, MongoDB for Document).
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from pymongo import MongoClient

# 1. MySQL Setup (Structured Data)
MYSQL_URL = os.getenv("MYSQL_URL", "mysql+pymysql://sma_user:sma_password@localhost:3306/sma_db")
mysql_engine = create_engine(
    MYSQL_URL, 
    pool_size=5, 
    max_overflow=10,
    pool_timeout=30,
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=mysql_engine)
Base = declarative_base()

def get_mysql_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 2. MongoDB Setup (Unstructured Data)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo_admin:mongo_password@localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sma_unstructured")

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]

def get_mongo_db():
    return mongo_db