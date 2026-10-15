# Predição em batch: aplica o modelo salvo por treino_regressao.py às
# amostras de outros meses (geradas por gerar_amostras.py) e grava no HDFS:
#   /resultados/predicoes_batch    -> uma linha por corrida (real x previsto)
#   /resultados/metricas_por_mes   -> RMSE, MAE e R2 de cada mês
# Os dois saem como um único CSV com cabeçalho, prontos para exportar com
# ./exportar_resultados.sh.
#
# Não treina nada: só carrega um pipeline já salvo no HDFS (/modelos).
#
# Execução:
#   docker exec -it spark-master /spark/bin/spark-submit \
#       --master spark://spark-master:7077 /app/predicao_batch.py
# Sem --meses, usa todos os meses com amostra de predição no HDFS.
# Para escolher meses: --meses 2 6 12 | outro modelo em vez do melhor: --modelo gbt

import argparse
from functools import reduce

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, abs as _abs, round as _round
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

from preparacao import (HDFS, MODELOS, caminho_amostra, caminho_modelo,
                        criar_atributos, escolher_meses)

parser = argparse.ArgumentParser()
parser.add_argument("--meses", type=int, nargs="+",
                    help="meses a prever (padrão: todos com amostra de predição no HDFS)")
parser.add_argument("--modelo", default="melhor", choices=["melhor"] + MODELOS,
                    help="pipeline salvo pelo treino a ser usado")
args = parser.parse_args()

SAIDA_PREDICOES = HDFS + "/resultados/predicoes_batch"
SAIDA_METRICAS = HDFS + "/resultados/metricas_por_mes"

spark = SparkSession.builder \
    .appName("PredicaoBatch_NYCTaxi") \
    .master("spark://spark-master:7077") \
    .config("spark.sql.session.timeZone", "UTC") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

metadados = spark.read.json(caminho_modelo("metadados")).collect()
melhor = metadados[0]["melhor_modelo"]
nome_modelo = melhor if args.modelo == "melhor" else \
    next(m["modelo"] for m in metadados if m["modelo"].lower() == args.modelo)
caminho = caminho_modelo("melhor" if args.modelo == "melhor" else "pipeline_" + args.modelo)
print(f"Modelo treinado com o mês {metadados[0]['mes_treino']:02d}. Melhor no treino: {melhor}")
print(f"Usando: {nome_modelo} ({caminho})")
modelo = PipelineModel.load(caminho)

args.meses = escolher_meses(spark, "predicao", args.meses)
print(f"Meses da predição: {args.meses}")

amostras = [spark.read.parquet(caminho_amostra("predicao", m)) for m in args.meses]
df = criar_atributos(reduce(DataFrame.unionByName, amostras))

predicoes = (modelo.transform(df)
    .select(
        "mes",
        "tpep_pickup_datetime",
        "trip_distance",
        "trip_duration_min",
        "passenger_count",
        "payment_label",
        "fare_amount",
        _round("prediction", 2).alias("tarifa_prevista"),
    )
    .withColumn("erro_abs", _round(_abs(col("tarifa_prevista") - col("fare_amount")), 2))
    .withColumn("modelo", lit(nome_modelo))
    .orderBy("mes", "tpep_pickup_datetime")
    .cache()
)

# Métricas por mês: mostram se o modelo treinado em um mês generaliza para os outros
avaliadores = {m: RegressionEvaluator(labelCol="fare_amount",
                                      predictionCol="tarifa_prevista",
                                      metricName=m)
               for m in ["rmse", "mae", "r2"]}
metricas = []
for mes in args.meses:
    pred_mes = predicoes.filter(col("mes") == mes)
    valores = {m: ev.evaluate(pred_mes) for m, ev in avaliadores.items()}
    metricas.append((nome_modelo, mes, pred_mes.count(),
                     valores["rmse"], valores["mae"], valores["r2"]))
    print(f"Mês {mes:02d}: RMSE={valores['rmse']:.3f} | MAE={valores['mae']:.3f} "
          f"| R2={valores['r2']:.3f}")

predicoes.coalesce(1).write.mode("overwrite") \
    .option("header", True) \
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss") \
    .csv(SAIDA_PREDICOES)

spark.createDataFrame(metricas, ["modelo", "mes", "corridas", "rmse", "mae", "r2"]) \
    .coalesce(1).write.mode("overwrite").option("header", True).csv(SAIDA_METRICAS)

print(f"\nPredições salvas em: {SAIDA_PREDICOES} ({predicoes.count():,} linhas)")
print(f"Métricas salvas em:  {SAIDA_METRICAS}")
print("Para trazer os CSVs para o Codespace: ./exportar_resultados.sh")

spark.stop()
