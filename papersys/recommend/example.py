"""推荐系统使用示例。

这个示例展示了如何使用新的推荐系统：
1. 先筛选ID
2. 在训练时提前采样背景数据
3. 最后才读取embedding
4. 大幅减少S3读取开销
"""

from pathlib import Path

from loguru import logger

from ..config import load_config, AppConfig
from ..database.manager import PaperManager
from .recommender import Recommender


def main():
    """运行推荐系统示例。"""
    # 1. 加载配置
    config_path = Path(__file__).parents[2] / "config.toml"
    app_config = load_config(AppConfig, config_path)

    logger.info("=" * 60)
    logger.info("推荐系统示例 - 优化S3读取")
    logger.info("=" * 60)

    # 2. 连接数据库
    manager = PaperManager(uri=app_config.database.uri)

    # 3. 创建推荐器
    recommender = Recommender(manager, app_config.recommend)

    # 4. 训练模型
    logger.info("\n" + "=" * 60)
    logger.info("步骤1: 训练推荐模型")
    logger.info("=" * 60)

    categories = app_config.paper.categories
    logger.info(f"使用类别: {categories}")

    try:
        recommender.fit(categories)
        logger.info("✅ 模型训练完成")
    except Exception as e:
        logger.error(f"❌ 模型训练失败: {e}")
        raise

    # 5. 预测和推荐
    logger.info("\n" + "=" * 60)
    logger.info("步骤2: 预测和推荐")
    logger.info("=" * 60)

    try:
        results = recommender.predict(
            categories=categories,
            last_n_days=7  # 最近7天的论文
        )

        if not results.is_empty():
            logger.info(f"\n预测结果概览:")
            logger.info(f"总论文数: {results.height}")
            logger.info(f"推荐论文数: {results.filter(pl.col('show') == 1).height}")

            # 显示推荐的论文
            recommended = results.filter(pl.col("show") == 1).sort("score", descending=True)
            logger.info(f"\n推荐论文列表 (按分数排序):")
            logger.info(recommended.select(["id", "score"]))

            return results
        else:
            logger.warning("没有推荐结果")
            return None

    except Exception as e:
        logger.error(f"❌ 预测失败: {e}")
        raise


if __name__ == "__main__":
    import polars as pl
    
    main()
