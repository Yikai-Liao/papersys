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
    
    @property
    def metadata_table(self):
        try:
            table = self.db[PAPER_METADATA_TABLE]
        except KeyError:
            logger.debug("Metadata table not found, creating a new one.")
            table = self.create_metadata_table()
        return table
    
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
        table.create_scalar_index(CATEGORIES, index_type="LABEL_LIST")
        table.create_scalar_index(UPDATE_DATE)

        return table
    
    def drop_metadata_table(self):
        try:
            self.db.drop_table(PAPER_METADATA_TABLE)
            logger.info("Dropped metadata table.")
        except Exception as e:
            logger.warning(f"Error dropping metadata table: {e}, no action taken.")
            

BATCH_SIZE = int(1e5)

def __upsert(table, df: pl.DataFrame, primary_key: str = ID):
    table\
      .merge_insert(primary_key)\
      .when_matched_update_all()\
      .when_not_matched_insert_all()\
      .execute(df)
    return table

def upsert(table, df: pl.DataFrame, primary_key: str = ID):
    for i in range(0, df.height, BATCH_SIZE):
        batch_df = df[i:i+BATCH_SIZE]
        table = __upsert(table, batch_df, primary_key)
    return table
    
def add(table, df: pl.DataFrame):
    for i in range(0, df.height, BATCH_SIZE):
        batch_df = df[i:i+BATCH_SIZE]
        table.add(batch_df)
    return table