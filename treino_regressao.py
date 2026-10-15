# Treino de modelos de regressão da tarifa (fare_amount) com Spark ML,
# baseado na seção "Regressão" do notebook unidade_5.ipynb.
#
# Compatível com as imagens bde2020 (Spark 3.3.0, Python 3.7).
# Execução:
#   docker exec -it spark-master /spark/bin/spark-submit \
#       --master spark://spark-master:7077 /treino_regressao.py

import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, hour, dayofweek, when, round as _round,
                                   unix_timestamp, to_timestamp, create_map, lit)
from itertools import chain
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (Imputer, StringIndexer, OneHotEncoder,
                                VectorAssembler, StandardScaler)
from pyspark.ml.regression import (LinearRegression, RandomForestRegressor,
                                   GBTRegressor)
from pyspark.ml.evaluation import RegressionEvaluator

HDFS = "hdfs://namenode:9000"
ARQUIVO_DADOS = HDFS + "/dados/nyc_tripdata_2024_sample_1M.csv"

docker cp nyc_tripdata_2024_sample_1M.csv namenode:/tmp/
docker exec namenode hdfs dfs -mkdir -p /dados
docker exec namenode hdfs dfs -put -f /tmp/nyc_tripdata_2024_sample_1M.csv /dados/
SAIDA_MODELO = HDFS + "/modelos/regressao_tarifa"
SAIDA_METRICAS = HDFS + "/resultados/regressao_metricas"
SAIDA_PREDICOES = HDFS + "/resultados/regressao_predicoes"

spark = SparkSession.builder \
    .appName("Unidade5_Regressao_NYCTaxi") \
    .master("spark://spark-master:7077") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------------------
# Leitura (tipos explícitos nas colunas usadas, sem depender do inferSchema)
# ------------------------------------------------------------------------
df = (spark.read.csv(ARQUIVO_DADOS, header=True)
    .withColumn("tpep_pickup_datetime", to_timestamp("tpep_pickup_datetime"))
    .withColumn("tpep_dropoff_datetime", to_timestamp("tpep_dropoff_datetime"))
    .withColumn("passenger_count", col("passenger_count").cast("double"))
    .withColumn("trip_distance", col("trip_distance").cast("double"))
    .withColumn("fare_amount", col("fare_amount").cast("double"))
    .withColumn("payment_type", col("payment_type").cast("int"))
)

# ------------------------------------------------------------------------
# Limpeza (mesmas regras do notebook)
# ------------------------------------------------------------------------
df_limpo = (df
    .filter(col("trip_distance") > 0)
    .filter(col("fare_amount") >= 2.50)              # tarifa mínima oficial
    .filter(col("fare_amount") <= 300)               # remove casos extremos
    .filter(col("tpep_dropoff_datetime") > col("tpep_pickup_datetime"))
    .filter(col("payment_type").isin([1, 2]))        # apenas cartão e dinheiro
    .dropDuplicates()
)

# ------------------------------------------------------------------------
# Engenharia de atributos
# ------------------------------------------------------------------------
mapa_pagamento = {1: "credito", 2: "dinheiro"}
mapping_expr = create_map(*[lit(x) for x in
    chain.from_iterable(mapa_pagamento.items())])

df_feat = (df_limpo
    .withColumn("trip_duration_min",
        _round((unix_timestamp("tpep_dropoff_datetime")
              - unix_timestamp("tpep_pickup_datetime")) / 60, 2))
    .withColumn("pickup_hour", hour("tpep_pickup_datetime"))
    .withColumn("eh_fim_de_semana",
        when(dayofweek("tpep_pickup_datetime").isin([1, 7]), 1.0).otherwise(0.0))
    .withColumn("payment_label", mapping_expr[col("payment_type")])
    .select("trip_distance", "trip_duration_min", "passenger_count",
            "pickup_hour", "eh_fim_de_semana", "payment_label", "fare_amount")
)

train_df, test_df = df_feat.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()
print(f"Treino: {train_df.count():,} | Teste: {test_df.count():,}")

# ------------------------------------------------------------------------
# Pipeline de preparação (fit APENAS no treino)
# O Imputer entrou no pipeline para que a mediana venha só do treino.
# ------------------------------------------------------------------------
num_cols = ["trip_distance", "trip_duration_min", "passengers",
            "pickup_hour", "eh_fim_de_semana"]

imputer = Imputer(inputCols=["passenger_count"], outputCols=["passengers"],
                  strategy="median")
indexer = StringIndexer(inputCol="payment_label", outputCol="payment_idx",
                        handleInvalid="skip")
encoder = OneHotEncoder(inputCol="payment_idx", outputCol="payment_vec")
assembler_num = VectorAssembler(inputCols=num_cols, outputCol="features_num")
scaler = StandardScaler(inputCol="features_num",
                        outputCol="features_num_scaled",
                        withMean=True, withStd=True)
assembler_final = VectorAssembler(
    inputCols=["features_num_scaled", "payment_vec"], outputCol="features")

prep_model = Pipeline(stages=[imputer, indexer, encoder, assembler_num,
                              scaler, assembler_final]).fit(train_df)

train_prep = prep_model.transform(train_df).select("features", "fare_amount").cache()
test_prep = prep_model.transform(test_df).select("features", "fare_amount").cache()

# ------------------------------------------------------------------------
# Treino e avaliação dos regressores
# ------------------------------------------------------------------------
modelos = {
    "LinearRegression": LinearRegression(featuresCol="features",
                                         labelCol="fare_amount"),
    "RandomForest": RandomForestRegressor(featuresCol="features",
                                          labelCol="fare_amount",
                                          numTrees=50, seed=42),
    "GBT": GBTRegressor(featuresCol="features", labelCol="fare_amount",
                        maxIter=50, seed=42),
}

avaliadores = {m: RegressionEvaluator(labelCol="fare_amount",
                                      predictionCol="prediction",
                                      metricName=m)
               for m in ["rmse", "mae", "r2"]}

resultados = []
treinados = {}
for nome, estimador in modelos.items():
    inicio = time.time()
    modelo = estimador.fit(train_prep)
    duracao = time.time() - inicio
    pred = modelo.transform(test_prep)
    metricas = {m: ev.evaluate(pred) for m, ev in avaliadores.items()}
    treinados[nome] = modelo
    resultados.append((nome, metricas["rmse"], metricas["mae"],
                       metricas["r2"], duracao))
    print(f"{nome:16} RMSE={metricas['rmse']:.3f} | MAE={metricas['mae']:.3f} "
          f"| R2={metricas['r2']:.3f} | treino em {duracao:.1f}s")

# Quais atributos mais influenciam a tarifa?
nomes_features = num_cols + ["payment_vec"]
importancias = treinados["RandomForest"].featureImportances.toArray()
print("\nImportância dos atributos (Random Forest):")
for nome, imp in sorted(zip(nomes_features, importancias),
                        key=lambda x: x[1], reverse=True):
    print(f"  {nome:20} {imp:.4f}")

# ------------------------------------------------------------------------
# Persistência no HDFS
# ------------------------------------------------------------------------
melhor_nome = min(resultados, key=lambda r: r[1])[0]
print(f"\nMelhor modelo (menor RMSE): {melhor_nome}")

# Modelo final = preparação + regressor, aplicável direto sobre dados brutos
modelo_final = PipelineModel(stages=prep_model.stages + [treinados[melhor_nome]])
modelo_final.write().overwrite().save(SAIDA_MODELO)

spark.createDataFrame(resultados,
                      ["modelo", "rmse", "mae", "r2", "tempo_treino_s"]) \
    .coalesce(1).write.mode("overwrite").option("header", True).csv(SAIDA_METRICAS)

modelo_final.transform(test_df) \
    .select("trip_distance", "trip_duration_min", "payment_label",
            "fare_amount", _round("prediction", 2).alias("prediction")) \
    .write.mode("overwrite").option("header", True).csv(SAIDA_PREDICOES)

print(f"Modelo salvo em:    {SAIDA_MODELO}")
print(f"Métricas salvas em: {SAIDA_METRICAS}")
print(f"Predições em:       {SAIDA_PREDICOES}")

spark.stop()
