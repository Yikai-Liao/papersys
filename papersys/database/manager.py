import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import lancedb
from pathlib import Path
from loguru import logger
from datetime import date
from .schema import PAPER_METADATA_SCHEMA, paper_embedding_schema
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
    
    @property
    def embedding_table(self):
        try:
            table = self.db[EMBEDDING_TABLE]
        except KeyError:
            logger.debug("Embedding table not found, creating a new one.")
            raise ValueError("Embedding table does not exist. Please create it with the desired dimension first.")
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
            
    def create_embedding_table(self, dim: int):
        """
        Create the embedding table in the database.
        """
        if dim <= 0:
            raise ValueError("Embedding dimension must be a positive integer.")
        if EMBEDDING_TABLE in self.db:
            existing_dim = self.embedding_dim
            if existing_dim != dim:
                raise ValueError(f"Embedding table already exists with dimension {existing_dim}, cannot create with dimension {dim}.")
        
        table = self.db.create_table(
            name = EMBEDDING_TABLE, 
            schema = paper_embedding_schema(dim),
            exist_ok=True
        )
        table.create_scalar_index(ID)
        return table
    
    def drop_embedding_table(self):
        try:
            self.db.drop_table(EMBEDDING_TABLE)
            logger.info("Dropped embedding table.")
        except Exception as e:
            logger.warning(f"Error dropping embedding table: {e}, no action taken.")

    @property
    def embedding_dim(self) -> int | None:
        if EMBEDDING_TABLE not in self.db:
            return None
        table = self.db[EMBEDDING_TABLE]
        schema = table.schema
        dim: int = schema.field(EMBEDDING_VECTOR).type.list_size
        return dim
    
    def unembeded_papers(
        self, categories: list[str], 
        columns: None|list[str],
        start: date | None = None,
        end: date | None = None
    ) -> pl.DataFrame:

        # 1) Only take the ID column from the embeddings table (Arrow), avoid loading/converting the entire table
        emb_ds = self.embedding_table.to_lance()
        emb_ids_tbl = emb_ds.to_table(columns=[ID])                # Only 1 column
        emb_ids_arr = emb_ids_tbl.column(ID).combine_chunks()

        # 2) Build time range filter expression (if start/end are provided)
        meta_ds = self.metadata_table.to_lance()
        time_filter = None
        if start is not None or end is not None:
            # Convert Python date to Arrow date32 scalar for comparison
            if start is not None:
                start_scalar = pa.scalar(start, type=pa.date32())
                # (publish_date >= start) | (update_date >= start)
                time_filter = (pc.field(PUBLISH_DATE) >= start_scalar) | (pc.field(UPDATE_DATE) >= start_scalar)
            if end is not None:
                end_scalar = pa.scalar(end, type=pa.date32())
                # (publish_date <= end) | (update_date <= end)
                end_cond = (pc.field(PUBLISH_DATE) <= end_scalar) | (pc.field(UPDATE_DATE) <= end_scalar)
                time_filter = end_cond if time_filter is None else (time_filter & end_cond)
        
        # 3) Only take ID + CATEGORIES columns from metadata, with time filter applied at Lance level
        meta_two_cols = meta_ds.to_table(
            columns=[ID, CATEGORIES],
            filter=time_filter
        )
        undembedd = pl.from_arrow(meta_two_cols)

        # 3) Use Polars to filter, select paper ID columns that match the categories
        filtered_ids_arr = (
            undembedd
            .filter(
                pl.col(CATEGORIES).list.eval(
                    pl.any_horizontal([pl.element().str.starts_with(cat) for cat in categories])
                ).list.any()
            )
            .select(pl.col(ID).cast(pl.Utf8))   # Explicitly cast to small string, convenient for aligning with schema.string
            .to_arrow()                         # -> pa.Table
            .column(0)                          # -> ChunkedArray
            .combine_chunks()                   # -> pa.Array
        )
        # Convert Large String to String
        filtered_ids_arr = pa.array(filtered_ids_arr, type=pa.string())
        
        # When filtered_ids_arr is large, disable scalar index for better performance
        use_scalar_index = True if len(filtered_ids_arr) < 10000 else False
        logger.debug(f"Filtered IDs count: {len(filtered_ids_arr)}, Embedded IDs count: {len(emb_ids_arr)}, use_scalar_index={use_scalar_index}")

        # 4) Push the 'unembedded' condition to Lance (IN filtered_ids but NOT IN embedded_ids)
        needs_expr = pc.is_in(pc.field(ID), filtered_ids_arr)
        not_emb_expr = pc.invert(pc.is_in(pc.field(ID), emb_ids_arr))
        final_expr = not_emb_expr & needs_expr

        # 5) Only take matching rows from Lance side (can specify columns to return, avoid pulling back large columns)
        out_tbl = meta_ds.to_table(
            filter=final_expr,
            use_scalar_index=use_scalar_index,
            columns=columns
        )
        return pl.from_arrow(out_tbl)
        

BATCH_SIZE = int(1e5)

def __upsert(table, df: pl.DataFrame, primary_key: str = ID):
    table\
      .merge_insert(primary_key)\
      .when_matched_update_all()\
      .when_not_matched_insert_all()\
      .execute(df)
    return table

def upsert(table, df: pl.DataFrame | pa.Table, primary_key: str = ID):
    # Get the number of rows depending on the type
    num_rows = df.height if isinstance(df, pl.DataFrame) else df.num_rows
    for i in range(0, num_rows, BATCH_SIZE):
        batch_df = df[i:i+BATCH_SIZE]
        table = __upsert(table, batch_df, primary_key)
    return table
    
def add(table, df: pl.DataFrame | pa.Table):
    num_rows = df.height if isinstance(df, pl.DataFrame) else df.num_rows
    for i in range(0, num_rows, BATCH_SIZE):
        batch_df = df[i:i+BATCH_SIZE]
        table.add(batch_df)
    return table