# Treino de modelos de regressão da tarifa (fare_amount) com Spark ML,
# baseado na seção "Regressão" do notebook unidade_5.ipynb.
#
# Usa a amostra do mês de treino gerada por gerar_amostras.py (setup.sh).
# Compatível com as imagens bde2020 (Spark 3.3.0, Python 3.7).
# Execução:
#   docker exec -it spark-master /spark/bin/spark-submit \
#       --master spark://spark-master:7077 /app/treino_regressao.py --mes 1

import argparse
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import round as _round
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (Imputer, StringIndexer, OneHotEncoder,
                                VectorAssembler, StandardScaler)
from pyspark.ml.regression import (LinearRegression, RandomForestRegressor,
                                   GBTRegressor)
from pyspark.ml.evaluation import RegressionEvaluator

from preparacao import (HDFS, COLUNAS_MODELO, caminho_amostra, caminho_modelo,
                        criar_atributos)

parser = argparse.ArgumentParser()
parser.add_argument("--mes", type=int, default=1, help="mês de treino (1-12)")
args = parser.parse_args()

SAIDA_METRICAS = HDFS + "/resultados/regressao_metricas"
SAIDA_PREDICOES = HDFS + "/resultados/regressao_predicoes"

spark = SparkSession.builder \
    .appName("Unidade5_Regressao_NYCTaxi") \
    .master("spark://spark-master:7077") \
    .config("spark.sql.session.timeZone", "UTC") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------------------
# Leitura da amostra (já limpa) e engenharia de atributos
# ------------------------------------------------------------------------
df_feat = criar_atributos(spark.read.parquet(caminho_amostra("treino", args.mes))) \
    .select(*COLUNAS_MODELO, "fare_amount")

train_df, test_df = df_feat.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()
print(f"Mês de treino: {args.mes:02d}")
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
    resultados.append((nome, args.mes, metricas["rmse"], metricas["mae"],
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
# Persistência no HDFS (/modelos) -- usada pela segunda etapa, predicao_batch.py
#   preparacao/            só a preparação ajustada no treino (Imputer ... Scaler)
#   pipeline_<modelo>/     preparação + regressor, um por modelo treinado
#   melhor/                o pipeline completo de menor RMSE
#   metadados/             mês de treino, melhor modelo, métricas e colunas
# ------------------------------------------------------------------------
melhor_nome = min(resultados, key=lambda r: r[2])[0]
print(f"\nMelhor modelo (menor RMSE): {melhor_nome}")

prep_model.write().overwrite().save(caminho_modelo("preparacao"))
print(f"Pipeline de preparação salvo em: {caminho_modelo('preparacao')}")

for nome, modelo in treinados.items():
    # Pipeline completo = preparação + regressor, aplicável direto sobre os dados
    pipeline_completo = PipelineModel(stages=prep_model.stages + [modelo])
    destinos = [caminho_modelo("pipeline_" + nome.lower())]
    if nome == melhor_nome:
        destinos.append(caminho_modelo("melhor"))
    for destino in destinos:
        pipeline_completo.write().overwrite().save(destino)
        print(f"Pipeline {nome} salvo em: {destino}")

spark.createDataFrame(
    [(args.mes, melhor_nome, nome, rmse, mae, r2, ",".join(COLUNAS_MODELO))
     for nome, _, rmse, mae, r2, _ in resultados],
    ["mes_treino", "melhor_modelo", "modelo", "rmse", "mae", "r2", "colunas_entrada"]) \
    .coalesce(1).write.mode("overwrite").json(caminho_modelo("metadados"))

spark.createDataFrame(resultados,
                      ["modelo", "mes_treino", "rmse", "mae", "r2", "tempo_treino_s"]) \
    .coalesce(1).write.mode("overwrite").option("header", True).csv(SAIDA_METRICAS)

PipelineModel.load(caminho_modelo("melhor")).transform(test_df) \
    .select("trip_distance", "trip_duration_min", "payment_label",
            "fare_amount", _round("prediction", 2).alias("prediction")) \
    .write.mode("overwrite").option("header", True).csv(SAIDA_PREDICOES)

print(f"Métricas salvas em: {SAIDA_METRICAS}")
print(f"Predições em:       {SAIDA_PREDICOES}")

spark.stop()
