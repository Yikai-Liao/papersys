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
    """推荐系统主类。"""

    def __init__(self, manager: PaperManager, config: RecommendConfig):
        """初始化推荐器。
        
        Args:
            manager: 数据库管理器
            config: 推荐配置
        """
        self.manager = manager
        self.config = config
        self.model = None

    def load_preference_data(self, categories: list[str]) -> pl.DataFrame:
        """加载偏好数据。
        
        优化策略：先筛选ID，最后才读取embedding，减少S3开销。
        
        Args:
            categories: 论文类别列表
        
        Returns:
            包含 id, preference, embedding 的 DataFrame
        """
        logger.info("加载偏好数据...")

        # 步骤1: 从preference表只读取ID和preference列
        pref_tbl = self.manager.preference_table.to_lance()
        pref_data = pref_tbl.to_table(columns=[ID, PREFERENCE])
        pref_df = pl.from_arrow(pref_data)

        if pref_df.is_empty():
            logger.warning("偏好表为空")
            return pl.DataFrame()

        pref_ids = pref_df.select(ID).to_series().to_list()
        logger.debug(f"偏好表中有 {len(pref_ids)} 条记录")

        # 步骤2: 从metadata表获取这些ID的类别信息（只读ID和CATEGORIES列）
        meta_tbl = self.manager.metadata_table.to_lance()
        meta_filter = pc.is_in(pc.field(ID), pa.array(pref_ids, type=pa.string()))
        meta_data = meta_tbl.to_table(columns=[ID, CATEGORIES], filter=meta_filter)
        meta_df = pl.from_arrow(meta_data)

        # 步骤3: 在Polars中过滤指定类别
        filter_condition = pl.col(CATEGORIES).list.eval(
            pl.any_horizontal(
                [pl.element().str.starts_with(cat) for cat in categories]
            )
        ).list.any()

        filtered_meta = meta_df.filter(filter_condition)

        # 步骤4: 获取过滤后的ID列表
        filtered_ids = filtered_meta.select(ID).to_series().to_list()

        if not filtered_ids:
            logger.warning(f"没有匹配类别 {categories} 的偏好数据")
            return pl.DataFrame()

        logger.debug(f"类别过滤后剩余 {len(filtered_ids)} 条记录")

        # 步骤5: 检查这些ID在embedding表中是否存在
        emb_tbl = self.manager.embedding_table.to_lance()
        emb_ids_data = emb_tbl.to_table(columns=[ID])
        emb_ids_set = set(emb_ids_data.column(ID).to_pylist())

        # 只保留有embedding的ID
        final_ids = [id for id in filtered_ids if id in emb_ids_set]

        if not final_ids:
            logger.warning("过滤后没有任何论文有embedding")
            return pl.DataFrame()

        logger.debug(f"有embedding的记录: {len(final_ids)} 条")

        # 步骤6: 现在才从embedding表读取这些ID的向量（关键优化点！）
        emb_filter = pc.is_in(pc.field(ID), pa.array(final_ids, type=pa.string()))
        emb_data = emb_tbl.to_table(filter=emb_filter)
        emb_df = pl.from_arrow(emb_data)

        # 步骤7: 合并数据
        result = (
            pref_df.filter(pl.col(ID).is_in(final_ids))
            .join(emb_df, on=ID, how="inner")
            .select(ID, PREFERENCE, EMBEDDING_VECTOR)
        )

        logger.info(f"✅ 加载了 {result.height} 条偏好数据（已优化S3读取）")
        return result

    def load_background_data(
        self,
        categories: list[str],
        exclude_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        sample_size: int | None = None,
    ) -> pl.DataFrame:
        """加载背景数据（未标注的论文）。
        
        优化策略：先筛选ID并采样，最后才读取embedding，大幅减少S3读取。
        
        Args:
            categories: 论文类别列表
            exclude_ids: 要排除的论文ID列表
            start_date: 开始日期
            end_date: 结束日期
            sample_size: 采样数量（如果为None，返回全部）
        
        Returns:
            包含 id, embedding 的 DataFrame
        """
        logger.info("加载背景数据...")

        # 步骤1: 先获取所有有embedding的论文ID（只读ID列）
        emb_ds = self.manager.embedding_table.to_lance()
        emb_ids_tbl = emb_ds.to_table(columns=[ID])
        emb_ids_arr = emb_ids_tbl.column(ID).combine_chunks()
        logger.debug(f"embedding表中有 {len(emb_ids_arr)} 条记录")

        # 步骤2: 从metadata表过滤（只读ID和CATEGORIES列）
        meta_ds = self.manager.metadata_table.to_lance()

        # 构建时间过滤器
        time_filter = None
        if start_date is not None or end_date is not None:
            if start_date is not None:
                start_scalar = pa.scalar(start_date, type=pa.date32())
                time_filter = pc.field(UPDATE_DATE) >= start_scalar
            if end_date is not None:
                end_scalar = pa.scalar(end_date, type=pa.date32())
                end_cond = pc.field(UPDATE_DATE) <= end_scalar
                time_filter = (
                    end_cond if time_filter is None else (time_filter & end_cond)
                )

        # 读取metadata（只读ID和CATEGORIES列）
        meta_two_cols = meta_ds.to_table(
            columns=[ID, CATEGORIES], filter=time_filter
        )
        meta_df = pl.from_arrow(meta_two_cols)
        logger.debug(f"metadata过滤后有 {meta_df.height} 条记录")

        # 步骤3: 在Polars中过滤指定类别
        filter_condition = pl.col(CATEGORIES).list.eval(
            pl.any_horizontal(
                [pl.element().str.starts_with(cat) for cat in categories]
            )
        ).list.any()

        filtered_ids_arr = (
            meta_df.filter(filter_condition)
            .select(pl.col(ID).cast(pl.Utf8))
            .to_arrow()
            .column(0)
            .combine_chunks()
        )

        # 转换为 String 类型
        filtered_ids_arr = pa.array(filtered_ids_arr, type=pa.string())
        logger.debug(f"类别过滤后有 {len(filtered_ids_arr)} 条记录")

        # 步骤4: 构建最终的ID列表：在类别中 & 有embedding & 不在排除列表
        # 使用Arrow的集合操作
        needs_expr = pc.is_in(pc.field(ID), filtered_ids_arr)
        has_emb_expr = pc.is_in(pc.field(ID), emb_ids_arr)
        
        # 从metadata获取满足条件的ID（在Arrow层面过滤）
        use_scalar_index = len(filtered_ids_arr) < 10000
        candidate_tbl = meta_ds.to_table(
            filter=needs_expr & has_emb_expr,
            use_scalar_index=use_scalar_index,
            columns=[ID]
        )
        
        candidate_ids = pl.from_arrow(candidate_tbl).select(ID).to_series().to_list()
        logger.debug(f"有embedding的候选记录: {len(candidate_ids)} 条")

        # 步骤5: 排除已标注的ID（在Python中处理，因为数量已经大幅减少）
        if exclude_ids:
            candidate_ids_set = set(candidate_ids) - set(exclude_ids)
            candidate_ids = list(candidate_ids_set)
            logger.debug(f"排除已标注后剩余: {len(candidate_ids)} 条")

        if not candidate_ids:
            logger.warning("没有符合条件的背景数据")
            return pl.DataFrame()

        # 步骤6: 【关键优化】在读取embedding之前先采样ID
        if sample_size is not None and sample_size < len(candidate_ids):
            import random
            random.seed(self.config.seed)
            sampled_ids = random.sample(candidate_ids, sample_size)
            logger.info(f"从 {len(candidate_ids)} 条候选数据中采样 {sample_size} 条")
        else:
            sampled_ids = candidate_ids
            logger.info(f"使用全部 {len(sampled_ids)} 条候选数据")

        # 步骤7: 最后才从embedding表读取采样后的ID的向量（大幅减少S3读取！）
        sampled_ids_arr = pa.array(sampled_ids, type=pa.string())
        emb_filter = pc.is_in(pc.field(ID), sampled_ids_arr)
        emb_data = emb_ds.to_table(filter=emb_filter)
        emb_df = pl.from_arrow(emb_data)

        logger.info(f"✅ 加载了 {emb_df.height} 条背景数据（已优化S3读取）")
        return emb_df.select(ID, EMBEDDING_VECTOR)

    def fit(self, categories: list[str]) -> "Recommender":
        """训练推荐模型。
        
        Args:
            categories: 论文类别列表
        
        Returns:
            自身（支持链式调用）
        """
        # 加载偏好数据
        pref_data = self.load_preference_data(categories)
        if pref_data.is_empty():
            raise ValueError("没有可用的偏好数据")

        # 计算需要的背景数据数量
        positive_count = pref_data.filter(pl.col(PREFERENCE) == "like").height
        sample_size = int(positive_count * self.config.neg_sample_ratio)
        logger.info(f"正样本数量: {positive_count}, 需要采样背景数据: {sample_size}")

        # 加载背景数据（排除已标注的，并在读取embedding前采样）
        pref_ids = pref_data.select(ID).to_series().to_list()
        background_data = self.load_background_data(
            categories=categories,
            exclude_ids=pref_ids,
            sample_size=sample_size  # 关键：在读取embedding前就确定数量
        )

        if background_data.is_empty():
            raise ValueError("没有可用的背景数据")

        # 准备训练数据
        pref_data = pref_data.rename({EMBEDDING_VECTOR: "embedding"})
        background_data = background_data.rename({EMBEDDING_VECTOR: "embedding"})

        # 训练模型
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
        """预测并推荐论文。
        
        优化策略：先筛选出所有目标ID，最后才读取embedding。
        
        Args:
            categories: 论文类别列表
            last_n_days: 最近N天的论文（与 start_date/end_date 互斥）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            包含 id, score, show 等列的 DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 fit() 方法")

        logger.info("开始预测和推荐...")

        # 确定时间范围
        predict_config = self.config.predict
        if last_n_days is None and start_date is None and end_date is None:
            last_n_days = predict_config.last_n_days

        if last_n_days is not None:
            end_date = date.today()
            start_date = end_date - timedelta(days=last_n_days)
            logger.info(f"使用最近{last_n_days}天的数据: {start_date} 到 {end_date}")

        # 加载目标数据（排除已标注的，不采样 - 预测时需要全部数据）
        pref_ids = (
            self.load_preference_data(categories).select(ID).to_series().to_list()
        )

        target_data = self.load_background_data(
            categories=categories,
            exclude_ids=pref_ids,
            start_date=start_date,
            end_date=end_date,
            sample_size=None  # 预测时不采样，需要全部数据
        )

        if target_data.is_empty():
            logger.warning("没有符合条件的目标数据")
            return pl.DataFrame()

        logger.info(f"目标数据: {target_data.height} 条")

        # 过滤 NaN 值
        target_data = self._filter_nan_embeddings(target_data)

        if target_data.is_empty():
            logger.warning("过滤 NaN 后没有数据")
            return pl.DataFrame()

        # 提取特征并预测
        X_target = np.vstack(target_data[EMBEDDING_VECTOR].to_numpy())
        X_target = np.nan_to_num(X_target, nan=0.0)

        try:
            scores = self.model.predict_proba(X_target)[:, 1]
            logger.info(
                f"预测完成，分数范围: {np.min(scores):.4f} - {np.max(scores):.4f}"
            )
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise

        # 自适应采样确定推荐
        show_flags = adaptive_sample(
            scores,
            target_sample_rate=predict_config.sample_rate,
            high_threshold=predict_config.high_threshold,
            boundary_threshold=predict_config.boundary_threshold,
            random_state=self.config.seed,
        )

        # 添加预测结果
        result = target_data.with_columns(
            [
                pl.lit(scores).alias("score"),
                pl.lit(show_flags.astype(np.int8)).alias("show"),
            ]
        ).drop(EMBEDDING_VECTOR)

        recommended_count = np.sum(show_flags)
        logger.info(
            f"推荐完成: 总计 {result.height} 篇论文中推荐 {recommended_count} 篇 "
            f"({recommended_count/result.height*100:.2f}%)"
        )

        return result

    def _filter_nan_embeddings(self, df: pl.DataFrame) -> pl.DataFrame:
        """过滤包含 NaN 的嵌入向量。"""
        logger.info("过滤包含 NaN 的嵌入向量...")

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
                f"过滤了 {removed_count}/{df.height} "
                f"({removed_count/df.height*100:.2f}%) 个含 NaN 的样本"
            )
            df = df.with_row_index("__idx__")
            valid_indices = np.where(~nan_mask)[0]
            df = df.filter(pl.col("__idx__").is_in(valid_indices)).drop("__idx__")
            logger.info(f"过滤后: {df.height} 条数据")
        else:
            logger.info("✅ 没有包含 NaN 的嵌入向量")

        return df
