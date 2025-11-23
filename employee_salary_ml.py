from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import happybase

# 1: Spark session with Hive support
spark = SparkSession.builder.appName("EmployeeSalaryML").enableHiveSupport().getOrCreate()

# 2: Load data from your Hive table
spark.sql("USE final_project")
df = spark.sql("SELECT Experience_Years, Age, Monthly_Salary FROM employee_salary")

# 3: Drop rows with nulls
df = df.na.drop(subset=["Experience_Years", "Age", "Monthly_Salary"])

# 4: Assemble features
assembler = VectorAssembler(
    inputCols=["Experience_Years", "Age"],
    outputCol="features"
)
assembled_df = assembler.transform(df).select("features", "Monthly_Salary")

# 5: Train/test split
train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)

# 6: Train Linear Regression model
lr = LinearRegression(labelCol="Monthly_Salary", featuresCol="features")
lr_model = lr.fit(train_data)

# 7: Evaluate model
predictions = lr_model.transform(test_data)

evaluator_rmse = RegressionEvaluator(
    labelCol="Monthly_Salary",
    predictionCol="prediction",
    metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="Monthly_Salary",
    predictionCol="prediction",
    metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# 8: Write metrics to HBase (employee_metrics, stats)
connection = happybase.Connection('localhost')
connection.open()
table = connection.table('employee_metrics')
table.put(
    b'metrics1',
    {
        b'stats:rmse': str(rmse).encode('utf-8'),
        b'stats:r2':   str(r2).encode('utf-8'),
    }
)
connection.close()

# 9: Stop Spark
spark.stop()