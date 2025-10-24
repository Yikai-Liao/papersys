import pyarrow as pa
import polars as pl
from typing import Dict

from .name import *

PAPER_METADATA_SCHEMA = pa.schema(
    [
        pa.field(ID, pa.string()),
        pa.field(SUBMITTER, pa.string()),
        pa.field(AUTHORS, pa.string()),
        pa.field(TITLE, pa.string()),
        pa.field(COMMENTS, pa.string(), nullable=True),
        pa.field(JOURNAL_REF, pa.string(), nullable=True),
        pa.field(DOI, pa.string(), nullable=True),
        pa.field(REPORT_NO, pa.string(), nullable=True),
        pa.field(CATEGORIES, pa.string(), nullable=True),
        pa.field(LICENSE, pa.string(), nullable=True),
        pa.field(ABSTRACT, pa.string()),
        pa.field(UPDATE_DATE, pa.date32()),
    ]
)


def pa_field_to_pl_dtype(field: pa.Field) -> pl.DataType:
    t = field.type
    # Handle primitive types
    if pa.types.is_int8(t) or pa.types.is_int16(t) or pa.types.is_int32(t) or pa.types.is_int64(t):
        return pl.Int64() if pa.types.is_int64(t) else pl.Int32() if pa.types.is_int32(t) else pl.Int16() if pa.types.is_int16(t) else pl.Int8()
    if pa.types.is_uint8(t) or pa.types.is_uint16(t) or pa.types.is_uint32(t) or pa.types.is_uint64(t):
        return pl.UInt64() if pa.types.is_uint64(t) else pl.UInt32() if pa.types.is_uint32(t) else pl.UInt16() if pa.types.is_uint16(t) else pl.UInt8()
    if pa.types.is_float16(t) or pa.types.is_float32(t) or pa.types.is_float64(t):
        return pl.Float64() if pa.types.is_float64(t) else pl.Float32() if pa.types.is_float32(t) else pl.Float32()
    if pa.types.is_boolean(t):
        return pl.Boolean()
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return pl.Utf8()
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return pl.Binary()
    if pa.types.is_dictionary(t):
        # map dictionary / categorical to pl.Categorical
        return pl.Categorical()
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        # For lists, get the nested field type and return List(dtype)
        value_field = t.value_field if hasattr(t, 'value_field') else t.value_type
        inner_dtype = pa_field_to_pl_dtype(value_field)
        return pl.List(inner_dtype)
    if pa.types.is_timestamp(t):
        # Preserve timezone info if present by using pl.Datetime(time_unit, timezone)
        unit = t.unit if hasattr(t, 'unit') else 'us'
        tz = t.tz if hasattr(t, 'tz') else None
        return pl.Datetime(time_unit=unit, timezone=tz) if tz else pl.Datetime(time_unit=unit)
    if pa.types.is_date32(t) or pa.types.is_date64(t):
        return pl.Date
    # Fallback: use Utf8 to avoid losing data
    return pl.Utf8()

def pa2pl_schema(pa_schema: pa.Schema) -> Dict[str, pl.DataType]:
    """Convert a pyarrow.Schema to a polars schema mapping name->pl.DataType."""
    out = {}
    for field in pa_schema:
        out[field.name] = pa_field_to_pl_dtype(field)
    return out
