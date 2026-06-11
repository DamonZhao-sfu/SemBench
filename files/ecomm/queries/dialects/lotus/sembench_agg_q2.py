import os
import time
from sem_ops_register import register_semantic_operators  # DO NOT remove this line
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg


parallelism = 1
jar_path = "/localhome/hza214/AI_UDF_private/sparkai/target/scala-2.12/spark-semantic-plugin_2.12-0.1.0-SNAPSHOT.jar"


def _build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("SemBenchAggQ2")
        .master("local[*]")
        .config("spark.driver.memory", "32g")
        .config("spark.executor.memory", "32g")
        .config("spark.jars", jar_path)
        .config("spark.sql.extensions", "com.huawei.sparkai.SemanticTopKPlugin,com.huawei.sparkai.SemanticMapPlugin,com.huawei.sparkai.SemanticJoinPlugin,com.huawei.sparkai.SemanticFilterPlugin,com.huawei.sparkai.SemanticAggregatePlugin,com.huawei.sparkai.SemanticGroupByPlugin,com.huawei.sparkai.ApproximateFilterPlugin,com.huawei.sparkai.ApproximateJoinPlugin")
        .config("spark.driver.extraClassPath", jar_path)
        .config("spark.executor.extraClassPath", jar_path)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.optimizer.columnPruning.enabled", "false")
        .config("spark.huawei.sparkai.disableFilterReorder", "true")
        .config("spark.default.parallelism", str(parallelism))
        .getOrCreate()
    )


def run(data_dir: str, scale_factor: int = 157376):
    spark = _build_spark()
    t0 = time.time()
    base_dir = data_dir if os.path.exists(os.path.join(data_dir, "styles_details.parquet")) else os.path.join(data_dir, "data", f"sf_{scale_factor}")
    styles = spark.read.parquet(os.path.join(base_dir, "styles_details.parquet"))
    filtered = styles.sem_filter(
        prompt="The product is a pair of running shoes. Product: {text:productDisplayName} {text:productDescriptors}",
        batch_size=128,
        model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        url="http://localhost:8000/v1/chat/completions",
        use_cascade=False,
        is_approximate=False,
        verbose=False,
        timeout=300,
    )
    result = filtered.select(avg("price").alias("avg_price_running_shoes"))
    pdf = result.toPandas()
    print(f"Q2 completed in {time.time() - t0:.2f}s")
    spark.stop()
    return pdf
