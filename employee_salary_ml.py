from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import happybase


def main():
    # 1. Start Spark with Hive support
    spark = (
        SparkSession.builder
        .appName("EmployeeSalaryML")
        .enableHiveSupport()
        .getOrCreate()
    )

    # 2. Use the Hive database and load data from the employee_salary table
    spark.sql("USE final_project")
    df = spark.sql(
        "SELECT Experience_Years, Age, Monthly_Salary "
        "FROM employee_salary"
    )

    # 3. Drop any rows with nulls in the columns we need
    df = df.na.drop(subset=["Experience_Years", "Age", "Monthly_Salary"])

    # 4. Assemble features into a single vector
    assembler = VectorAssembler(
        inputCols=["Experience_Years", "Age"],
        outputCol="features"
    )
    data = assembler.transform(df).select("features", "Monthly_Salary")

    # 5. Split into train and test sets
    train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

    # 6. Train a Linear Regression model to predict Monthly_Salary
    lr = LinearRegression(
        labelCol="Monthly_Salary",
        featuresCol="features"
    )
    lr_model = lr.fit(train_data)

    # 7. Evaluate the model on the test data
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

    print("=== Evaluation Metrics ===")
    print(f"RMSE: {rmse}")
    print(f"R2:   {r2}")

    # 8. Write metrics to HBase (employee_metrics table, stats column family)
    connection = happybase.Connection("localhost")
    connection.open()
    table = connection.table("employee_metrics")

    row_key = b"run1"  # row key for this model run
    table.put(
        row_key,
        {
            b"stats:rmse": str(rmse).encode("utf-8"),
            b"stats:r2":   str(r2).encode("utf-8"),
        },
    )
    connection.close()

    # 9. Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()