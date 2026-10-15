# Gera, a partir dos arquivos mensais completos da TLC no HDFS, as amostras
# usadas no treino e na predição em batch. Chamado pelo setup.sh.
#
# As amostras são tiradas DEPOIS da limpeza, então todas as linhas são
# corridas válidas, e saem em Parquet no HDFS (/dados/amostras).
#
# Execução:
#   docker exec -it spark-master /spark/bin/spark-submit \
#       --master spark://spark-master:7077 /app/gerar_amostras.py \
#       --mes-treino 1 --n-treino 200000 --meses 2 6 12 --n-predicao 10000

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

from preparacao import ler_mes, limpar, caminho_amostra

parser = argparse.ArgumentParser()
parser.add_argument("--mes-treino", type=int, default=1)
parser.add_argument("--n-treino", type=int, default=200000)
parser.add_argument("--meses", type=int, nargs="+", default=[2, 6, 12],
                    help="meses da predição em batch")
parser.add_argument("--n-predicao", type=int, default=10000)
args = parser.parse_args()

spark = SparkSession.builder \
    .appName("GerarAmostras_NYCTaxi") \
    .master("spark://spark-master:7077") \
    .config("spark.sql.session.timeZone", "UTC") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")


def gerar(tipo, mes, n):
    df = limpar(ler_mes(spark, mes))
    total = df.count()
    # sample() reduz o volume antes do embaralhamento; a margem de 20% garante
    # linhas suficientes e o limit() fixa o tamanho exato da amostra.
    fracao = min(1.0, n * 1.2 / total) if total else 1.0
    amostra = (df.sample(fraction=fracao, seed=42)
                 .orderBy(rand(seed=42))
                 .limit(n))
    destino = caminho_amostra(tipo, mes)
    amostra.write.mode("overwrite").parquet(destino)
    print(f"{tipo:8} mês {mes:02d}: {min(n, total):,} de {total:,} corridas válidas -> {destino}")


gerar("treino", args.mes_treino, args.n_treino)
for mes in args.meses:
    gerar("predicao", mes, args.n_predicao)

spark.stop()
