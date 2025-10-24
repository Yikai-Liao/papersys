import polars as pl
import lancedb
from pathlib import Path
from loguru import logger
from .schema import PAPER_METADATA_SCHEMA
from .name import *


class PaperManager:
    def __init__(self, uri: str = "data/sample-lancedb"):
        logger.info(f"Connecting to database at {uri}")
        self.db = lancedb.connect(uri)
    
    def create_metadata_table(self):
        """
        Create the metadata table in the database.
        """
        table = self.db.create_table(
            name = PAPER_METADATA_TABLE, 
            schema = PAPER_METADATA_SCHEMA,
            exist_ok=True
        )

        table.create_scalar_index(ID)
        table.create_scalar_index(CATEGORIES)
        table.create_scalar_index(UPDATE_DATE)

        return table
    
    def drop_metadata_table(self):
        try:
            self.db.drop_table(PAPER_METADATA_TABLE)
            logger.info("Dropped metadata table.")
        except Exception as e:
            logger.error(f"Error dropping metadata table: {e}")

    def add_metadata(self, df: pl.DataFrame):
        try:
            table = self.db[PAPER_METADATA_TABLE]
        except KeyError:
            logger.debug("Metadata table not found, creating a new one.")
            table = self.create_metadata_table()
            
        table.add(df)
        