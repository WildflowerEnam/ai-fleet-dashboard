"""
    The single place that knows how to connect to the database.
    Every other file in the project imports 'sessionLocal' from here

    This file:
        - reads the database URL from the DATABASE_URL environment variable 
        - creates one persistent engine (a connection pool)
        - exposes a SessionLocal() factory for short-lived sessions

"""

import os 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker 


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+pysycopg2://fleet:fleet@postgres:5432/fleet",
)

# engine 
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

# session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    future=True,
)

