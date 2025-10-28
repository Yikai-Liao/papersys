"""Main recommendation logic."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from ..config import RecommendConfig
from ..database.manager import PaperManager
from ..database.name import CATEGORIES, EMBEDDING_VECTOR, ID, PREFERENCE, UPDATE_DATE
from .sampler import adaptive_sample
from .trainer import train_model


class Recommender:
    """Main recommender class."""

    def __init__(self, manager: PaperManager, config: RecommendConfig):
        """Initialize the recommender.
        
        Args:
            manager: Database manager
            config: Recommendation config
        """
        self.manager = manager
        self.config = config
        self.model = None
        self._pref_cache: pl.DataFrame | None = None
        self._pref_cache_key: tuple[str, ...] | None = None
        self._pref_ids: list[str] = []
        self._embedding_ids_cache: pa.Array | None = None
        self._categories_key: tuple[str, ...] | None = None
        self._meta_cache: dict[tuple, pl.DataFrame] = {}

    def _categories_filter_expr(self, categories: list[str]) -> pl.Expr:
        """Build a Polars expression to filter by category prefixes."""
        if not categories:
            return pl.lit(True)

        prefix_checks = [pl.element().str.starts_with(cat) for cat in categories]
        return (
            pl.col(CATEGORIES)
            .list.eval(pl.any_horizontal(prefix_checks))
            .list.any()
        )

    def _build_time_filter(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> "pc.Expression | None":
        """Build Arrow filter expression based on update_date."""
        time_filter = None

        if start_date is not None:
            start_scalar = pa.scalar(start_date, type=pa.date32())
            time_filter = pc.field(UPDATE_DATE) >= start_scalar

        if end_date is not None:
            end_scalar = pa.scalar(end_date, type=pa.date32())
            end_expr = pc.field(UPDATE_DATE) <= end_scalar
            time_filter = end_expr if time_filter is None else (time_filter & end_expr)

        return time_filter

    def _embedding_id_array(self) -> pa.Array:
        """Return all IDs from the embedding table as an Arrow array."""
        if self._embedding_ids_cache is not None:
            return self._embedding_ids_cache

        emb_tbl = self.manager.embedding_table.to_lance()
        emb_ids_col = emb_tbl.to_table(columns=[ID]).column(ID)
        self._embedding_ids_cache = emb_ids_col.combine_chunks()
        return self._embedding_ids_cache

    def _metadata_subset(
        self,
        categories: list[str],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Get cached metadata subset, already filtered by categories and presence of embeddings."""

        key = (tuple(sorted(categories)), start_date, end_date)
        cached = self._meta_cache.get(key)
        if cached is not None:
            return cached

        meta_tbl = self.manager.metadata_table.to_lance()
        filter_expr = self._build_time_filter(start_date, end_date)

        embedding_ids = self._embedding_id_array()
        emb_expr = pc.is_in(pc.field(ID), embedding_ids)
        combined_expr = emb_expr if filter_expr is None else (filter_expr & emb_expr)

        use_scalar_index = len(embedding_ids) < 10_000

        meta_arrow = meta_tbl.to_table(
            columns=[ID, CATEGORIES],
            filter=combined_expr,
            prefilter=True,
            use_scalar_index=use_scalar_index,
        )
        meta_df = pl.from_arrow(meta_arrow)

        if meta_df.is_empty():
            self._meta_cache[key] = meta_df
            return meta_df

        filtered = meta_df.filter(self._categories_filter_expr(categories))
        self._meta_cache[key] = filtered
        return filtered

    def _metadata_by_categories(
        self,
        categories: list[str],
        *,
        restrict_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Apply additional filtering on top of the cached metadata subset."""

        meta_df = self._metadata_subset(
            categories,
            start_date=start_date,
            end_date=end_date,
        )

        if meta_df.is_empty() or not restrict_ids:
            return meta_df

        return meta_df.filter(pl.col(ID).is_in(restrict_ids))

    def _cast_embeddings_to_float32(self, table: pa.Table) -> pa.Table:
        """Cast embedding column from float16 to float32 (Arrow) before converting to Polars."""

        idx = table.schema.get_field_index(EMBEDDING_VECTOR)
        if idx == -1:
            return table

        vector_field = table.schema.field(idx)
        vector_type = vector_field.type

        if not (
            pa.types.is_list(vector_type) or pa.types.is_fixed_size_list(vector_type)
        ):
            return table

        value_type = vector_type.value_type
        if not pa.types.is_float16(value_type):
            return table

        if pa.types.is_fixed_size_list(vector_type):
            target_type = pa.list_(pa.float32(), vector_type.list_size)
        else:
            target_type = pa.list_(pa.float32())
        casted = pc.cast(table.column(idx), target_type)
        new_field = pa.field(EMBEDDING_VECTOR, target_type, vector_field.nullable)

        return table.set_column(idx, new_field, casted)

    def _load_embeddings_for_ids(self, ids: list[str]) -> pl.DataFrame:
        """Load embeddings by ID from the embedding table."""

        if not ids:
            return pl.DataFrame()

        emb_tbl = self.manager.embedding_table.to_lance()
        id_array = pa.array(ids, type=pa.string())
        filter_expr = pc.is_in(pc.field(ID), id_array)
        use_scalar_index = len(ids) < 10000

        emb_arrow = emb_tbl.to_table(
            filter=filter_expr,
            prefilter=True,
            use_scalar_index=use_scalar_index,
            columns=[ID, EMBEDDING_VECTOR],
        )

        emb_arrow = self._cast_embeddings_to_float32(emb_arrow)
        return pl.from_arrow(emb_arrow)

    def load_preference_data(
        self,
        categories: list[str],
        *,
        refresh: bool = False,
    ) -> pl.DataFrame:
        """Load preference data.
        
        Strategy: filter candidate IDs first, delay loading embeddings to reduce S3 IO.
        
        Args:
            categories: Paper categories
            refresh: Bypass cache and reload
        
        Returns:
            DataFrame containing id, preference, embedding
        """
        key = tuple(sorted(categories))
        meta_key = (key, None, None)
        if (
            not refresh
            and self._pref_cache is not None
            and self._pref_cache_key == key
        ):
            logger.info("Using cached preference data: {} rows", self._pref_cache.height)
            return self._pref_cache

        logger.info("Loading preference data...")

        if refresh:
            self._meta_cache.pop(meta_key, None)

        pref_tbl = self.manager.preference_table.to_lance()
        pref_arrow = pref_tbl.to_table(columns=[ID, PREFERENCE])
        pref_df = pl.from_arrow(pref_arrow)

        if pref_df.is_empty():
            logger.warning("Preference table is empty")
            return pl.DataFrame()

        pref_ids = pref_df.select(ID).to_series().to_list()
        logger.debug(f"Preference table has {len(pref_ids)} rows")

        meta_df = self._metadata_by_categories(
            categories,
            restrict_ids=pref_ids,
        )

        filtered_ids = meta_df.select(ID).to_series().to_list()
        if not filtered_ids:
            logger.warning(f"No preference data matching categories {categories}")
            return pl.DataFrame()

        logger.debug(f"After category filter: {len(filtered_ids)} rows")

        emb_df = self._load_embeddings_for_ids(filtered_ids)

        result = (
            pref_df.filter(pl.col(ID).is_in(filtered_ids))
            .join(emb_df, on=ID, how="inner")
            .select(ID, PREFERENCE, EMBEDDING_VECTOR)
        )

        logger.info(f"✅ Loaded {result.height} preference rows")

        self._pref_cache = result
        self._pref_cache_key = key
        self._pref_ids = result.select(ID).to_series().to_list()

        return result

    def _load_candidate_embeddings(
        self,
        *,
        categories: list[str],
        purpose: str,
        exclude_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        sample_size: int | None = None,
    ) -> pl.DataFrame:
        """Load candidate paper embeddings."""

        logger.info("Loading {} data...", purpose)

        meta_df = self._metadata_by_categories(
            categories,
            start_date=start_date,
            end_date=end_date,
        )

        logger.debug(f"Metadata after filter: {meta_df.height} rows")

        if meta_df.is_empty():
            logger.warning("No eligible {} data", purpose)
            return pl.DataFrame()

        candidate_df = meta_df
        if exclude_ids:
            candidate_df = candidate_df.filter(~pl.col(ID).is_in(exclude_ids))
            logger.debug(f"After excluding labeled: {candidate_df.height} rows")

        if candidate_df.is_empty():
            logger.warning("No eligible {} data", purpose)
            return pl.DataFrame()

        candidate_ids = candidate_df.select(ID).to_series().to_list()
        logger.debug(f"Candidate rows with embeddings: {len(candidate_ids)}")

        if sample_size is not None and sample_size < len(candidate_ids):
            import random

            random.seed(self.config.seed)
            sampled_ids = random.sample(candidate_ids, sample_size)
            logger.info(f"Sampled {sample_size} from {len(candidate_ids)} for {purpose}")
        else:
            sampled_ids = candidate_ids
            logger.info(f"Using all {len(sampled_ids)} candidates for {purpose}")

        emb_df = self._load_embeddings_for_ids(sampled_ids)

        logger.info(f"✅ Loaded {emb_df.height} {purpose} rows")
        return emb_df.select(ID, EMBEDDING_VECTOR)

    def load_training_background(
        self,
        categories: list[str],
        exclude_ids: list[str] | None,
        sample_size: int | None,
    ) -> pl.DataFrame:
        """Load negative-sample candidates for training."""

        return self._load_candidate_embeddings(
            categories=categories,
            purpose="training background",
            exclude_ids=exclude_ids,
            sample_size=sample_size,
        )

    def load_prediction_candidates(
        self,
        categories: list[str],
        exclude_ids: list[str] | None,
        start_date: date | None,
        end_date: date | None,
    ) -> pl.DataFrame:
        """Load candidate paper embeddings for prediction."""

        return self._load_candidate_embeddings(
            categories=categories,
            purpose="prediction candidates",
            exclude_ids=exclude_ids,
            start_date=start_date,
            end_date=end_date,
        )

    def fit(self, categories: list[str]) -> "Recommender":
        """Train the recommendation model.
        
        Args:
            categories: Paper categories
        
        Returns:
            Self (for chaining)
        """
        # Record current category key for cache reuse across fit/predict
        self._categories_key = tuple(sorted(categories))

        # Load preference data (cached)
        pref_data = self.load_preference_data(categories)
        if pref_data.is_empty():
            raise ValueError("No available preference data")

        # Compute required background sample size
        positive_count = pref_data.filter(pl.col(PREFERENCE) == "like").height
        sample_size = int(positive_count * self.config.neg_sample_ratio)
        logger.info(f"Positive samples: {positive_count}, negative sampling: {sample_size}")

        # Load background (exclude labeled; sample IDs before reading embeddings)
        background_data = self.load_training_background(
            categories=categories,
            exclude_ids=self._pref_ids,
            sample_size=sample_size,  # decide count before reading embeddings
        )

        if background_data.is_empty():
            raise ValueError("No available background data")

        # Prepare training data
        pref_data = pref_data.rename({EMBEDDING_VECTOR: "embedding"})
        background_data = background_data.rename({EMBEDDING_VECTOR: "embedding"})

        # Train model
        self.model = train_model(
            prefered_df=pref_data,
            remaining_df=background_data,
            embedding_columns=["embedding"],
            config=self.config,
        )

        return self

    def predict(
        self,
        categories: list[str],
        last_n_days: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Predict and recommend papers.
        
        Strategy: determine target IDs first, then load embeddings.
        
        Args:
            categories: Paper categories
            last_n_days: Use papers in the last N days (mutually exclusive with start/end)
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with id, score, show columns
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        logger.info("Starting prediction and recommendation...")

        # Resolve time window
        predict_config = self.config.predict
        if last_n_days is None and start_date is None and end_date is None:
            last_n_days = predict_config.last_n_days

        if last_n_days is not None:
            end_date = date.today()
            start_date = end_date - timedelta(days=last_n_days)
            logger.info(f"Using last {last_n_days} days: {start_date} to {end_date}")

        categories_key = tuple(sorted(categories))
        if self._pref_cache is None or self._pref_cache_key != categories_key:
            logger.debug("Categories differ from training; reloading preference data")
            self.load_preference_data(categories, refresh=True)

        pref_ids = self._pref_ids

        target_data = self.load_prediction_candidates(
            categories=categories,
            exclude_ids=pref_ids,
            start_date=start_date,
            end_date=end_date,
        )

        if target_data.is_empty():
            logger.warning("No eligible target data")
            return pl.DataFrame()

        logger.info(f"Target data: {target_data.height} rows")

        # Filter NaN values
        target_data = self._filter_nan_embeddings(target_data)

        if target_data.is_empty():
            logger.warning("No data after filtering NaN")
            return pl.DataFrame()

        # Extract features and predict
        X_target = np.vstack(target_data[EMBEDDING_VECTOR].to_numpy())
        X_target = np.nan_to_num(X_target, nan=0.0)

        try:
            scores = self.model.predict_proba(X_target)[:, 1]
            logger.info(
                f"Prediction complete. Score range: {np.min(scores):.4f} - {np.max(scores):.4f}"
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

        # Adaptive sampling to determine recommendations
        show_flags = adaptive_sample(
            scores,
            target_sample_rate=predict_config.sample_rate,
            high_threshold=predict_config.high_threshold,
            boundary_threshold=predict_config.boundary_threshold,
            random_state=self.config.seed,
        )

        # Attach prediction results
        result = target_data.with_columns(
            [
                pl.lit(scores).alias("score"),
                pl.lit(show_flags.astype(np.int8)).alias("show"),
            ]
        ).drop(EMBEDDING_VECTOR)

        recommended_count = np.sum(show_flags)
        logger.info(
            f"Recommendation complete: {recommended_count}/{result.height} "
            f"({recommended_count/result.height*100:.2f}%)"
        )

        return result

    def _filter_nan_embeddings(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter rows whose embedding vectors contain NaN."""
        logger.info("Filtering embeddings containing NaN...")

        nan_mask = np.zeros(df.height, dtype=bool)
        col_data = df[EMBEDDING_VECTOR].to_list()
        for i, vec in enumerate(col_data):
            if vec is None or (
                isinstance(vec, (list, np.ndarray)) and np.isnan(vec).any()
            ):
                nan_mask[i] = True

        removed_count = nan_mask.sum()
        if removed_count > 0:
            logger.warning(
                f"Removed {removed_count}/{df.height} "
                f"({removed_count/df.height*100:.2f}%) samples containing NaN"
            )
            df = df.with_row_index("__idx__")
            valid_indices = np.where(~nan_mask)[0]
            df = df.filter(pl.col("__idx__").is_in(valid_indices)).drop("__idx__")
            logger.info(f"After filtering: {df.height} rows")
        else:
            logger.info("✅ No embeddings contain NaN")

        return df
