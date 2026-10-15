# Leitura, limpeza e engenharia de atributos compartilhadas entre o treino
# (treino_regressao.py) e a predição em batch (predicao_batch.py) -- os dois
# precisam preparar os dados exatamente do mesmo jeito.
#
# Compatível com as imagens bde2020 (Spark 3.3.0, Python 3.7).

from itertools import chain

from pyspark.sql.functions import (col, hour, dayofweek, month, year, when,
                                   round as _round, unix_timestamp,
                                   create_map, lit)

HDFS = "hdfs://namenode:9000"
DIR_TLC = HDFS + "/dados/tlc"
DIR_AMOSTRAS = HDFS + "/dados/amostras"
DIR_MODELOS = HDFS + "/modelos"

# Modelos treinados por treino_regressao.py (cada um salvo como pipeline_<nome>)
MODELOS = ["linearregression", "randomforest", "gbt"]
ANO = 2024

# Colunas que o PipelineModel salvo espera receber
COLUNAS_MODELO = ["trip_distance", "trip_duration_min", "passenger_count",
                  "pickup_hour", "eh_fim_de_semana", "payment_label"]

MAPA_PAGAMENTO = {1: "credito", 2: "dinheiro"}


def caminho_mes(mes):
    return f"{DIR_TLC}/yellow_tripdata_{ANO}-{mes:02d}.parquet"


def caminho_amostra(tipo, mes):
    """tipo: "treino" ou "predicao" -- amostras geradas por gerar_amostras.py."""
    return f"{DIR_AMOSTRAS}/{tipo}_{ANO}-{mes:02d}"


def caminho_modelo(nome):
    """nome: "preparacao", "pipeline_<modelo>", "melhor" ou "metadados"."""
    return f"{DIR_MODELOS}/{nome}"


def ler_mes(spark, mes):
    """Lê o arquivo mensal da TLC, padroniza os tipos (variam entre meses) e
    descarta corridas com data fora do mês -- os arquivos oficiais trazem
    alguns registros com datas erradas (ex: 2002, 2009)."""
    df = spark.read.parquet(caminho_mes(mes))
    return (df
        .select(
            col("tpep_pickup_datetime").cast("timestamp"),
            col("tpep_dropoff_datetime").cast("timestamp"),
            col("passenger_count").cast("double"),
            col("trip_distance").cast("double"),
            col("fare_amount").cast("double"),
            col("payment_type").cast("int"),
        )
        .filter((year("tpep_pickup_datetime") == ANO)
                & (month("tpep_pickup_datetime") == mes))
        .withColumn("mes", lit(mes))
    )


def limpar(df):
    """Mesmas regras de limpeza do notebook unidade_5.ipynb."""
    return (df
        .filter(col("trip_distance") > 0)
        .filter(col("fare_amount") >= 2.50)              # tarifa mínima oficial
        .filter(col("fare_amount") <= 300)               # remove casos extremos
        .filter(col("tpep_dropoff_datetime") > col("tpep_pickup_datetime"))
        .filter(col("payment_type").isin([1, 2]))        # apenas cartão e dinheiro
    )


def criar_atributos(df):
    mapping_expr = create_map(*[lit(x) for x in
        chain.from_iterable(MAPA_PAGAMENTO.items())])
    return (df
        .withColumn("trip_duration_min",
            _round((unix_timestamp("tpep_dropoff_datetime")
                  - unix_timestamp("tpep_pickup_datetime")) / 60, 2))
        .withColumn("pickup_hour", hour("tpep_pickup_datetime"))
        .withColumn("eh_fim_de_semana",
            when(dayofweek("tpep_pickup_datetime").isin([1, 7]), 1.0).otherwise(0.0))
        .withColumn("payment_label", mapping_expr[col("payment_type")])
    )
