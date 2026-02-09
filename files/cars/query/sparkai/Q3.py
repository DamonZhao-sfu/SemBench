import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Configuration
jar_path = "/scratch/hpc-prf-haqc/haikai/AI_UDF_private/sparkai/target/scala-2.12/spark-semantic-plugin_2.12-0.1.0-SNAPSHOT.jar"

# Initialize Spark Session
spark = SparkSession.builder.appName("SemFilterImageDamage") \
    .master("local[*]") \
    .config("spark.driver.memory", "32g") \
    .config("spark.executor.memory", "32g") \
    .config("spark.jars", jar_path) \
    .config("spark.sql.extensions", "com.huawei.sparkai.SemanticTopKPlugin,com.huawei.sparkai.SemanticMapPlugin,com.huawei.sparkai.SemanticJoinPlugin,com.huawei.sparkai.SemanticFilterPlugin,com.huawei.sparkai.SemanticAggregatePlugin,com.huawei.sparkai.SemanticGroupByPlugin") \
    .config("spark.driver.extraClassPath", jar_path) \
    .config("spark.executor.extraClassPath", jar_path) \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.optimizer.columnPruning.enabled", "false") \
    .config("spark.huawei.sparkai.disableFilterReorder", "true") \
    .config("spark.default.parallelism", "1") \
    .getOrCreate()

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
scale_factor = 9836
data_base = "/scratch/hpc-prf-haqc/haikai/SemBench/files/cars"

car_data_path = os.path.join(data_base, "data", f"sf_{scale_factor}", f"car_data_{scale_factor}.csv")
image_data_path = os.path.join(data_base, "data", f"sf_{scale_factor}", f"image_car_data_{scale_factor}.csv")

car_data = spark.read.option("header", "true").option("inferSchema", "true").csv(car_data_path)
image_data = spark.read.option("header", "true").option("inferSchema", "true").csv(image_data_path)

start_time = time.time()

# ---------------------------------------------------------
# 2. Join cars with images on car_id
# ---------------------------------------------------------
joined_df = car_data.join(image_data, on="car_id", how="inner")

# ---------------------------------------------------------
# 3. Filter for Manual transmission
# ---------------------------------------------------------
joined_df = joined_df.filter(col("transmission") == "Manual")

# ---------------------------------------------------------
# 4. Semantic filter: car is NOT damaged (based on image)
# ---------------------------------------------------------
filtered_df = joined_df.sem_filter(
    prompt="You are given an image of a vehicle or its parts. Return true if car is not damaged. Image: {image:image_path}",
    batch_size=1024,
    model="Qwen/Qwen3-VL-30B-A3B-Instruct",
    url="http://localhost:8000/v1/chat/completions",
    verbose=False,
    timeout=300
)
filtered_df.cache()

# ---------------------------------------------------------
# 5. Select vin and limit to 10
# ---------------------------------------------------------
final_df = filtered_df.select("vin").limit(10)

# ---------------------------------------------------------
# 6. Output
# ---------------------------------------------------------
output_path = f"./car_sf{scale_factor}_q3_result.csv"

final_df.show(truncate=False)
final_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

duration = time.time() - start_time
print(f"Process completed in {duration:.2f} seconds.")

spark.stop()
