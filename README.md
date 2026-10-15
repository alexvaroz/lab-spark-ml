# Laboratório — Regressão com Spark ML em cluster Hadoop + Spark

Treina modelos de regressão para prever a tarifa (`fare_amount`) das corridas de táxi de Nova York (2024) usando **Spark ML**, em um cluster real Hadoop (HDFS) + Spark Standalone rodando em containers no **GitHub Codespaces**.

É a continuação do laboratório de processamento distribuído: usa as mesmas imagens `bde2020` (Spark 3.3.0, Hadoop 3) e a seção de regressão do notebook `unidade_5.ipynb`.

## Estrutura

| Arquivo | Função |
|---|---|
| `.devcontainer/devcontainer.json` | Codespace Ubuntu com Docker-in-Docker |
| `setup.sh` | Sobe o cluster, instala o numpy, envia a base ao HDFS e copia o script |
| `treino_regressao.py` | Limpeza, atributos, pipeline e treino de 3 regressores |

## Executando

1. Abra o repositório em um Codespace (**Code → Codespaces → Create codespace**).
2. No terminal:

   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Treine os modelos:

   ```bash
   docker exec -it spark-master /spark/bin/spark-submit \
       --master spark://spark-master:7077 /treino_regressao.py
   ```

   Acompanhe nas portas **8080** (Spark Master) e **4040** (estágios do job, só durante a execução).

## O que o script faz

1. Lê `/dados/nyc_tripdata_2024_sample_1M.csv` do HDFS.
2. Aplica as regras de limpeza do notebook: distância > 0, tarifa entre $2,50 e $300, desembarque depois do embarque, pagamento em cartão ou dinheiro.
3. Cria os atributos: duração da corrida, hora do embarque, fim de semana e tipo de pagamento.
4. Separa treino e teste (80/20, `seed=42`) e ajusta o pipeline de preparação só no treino (Imputer → StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler).
5. Treina **LinearRegression**, **RandomForestRegressor** e **GBTRegressor** e compara RMSE, MAE e R².
6. Salva no HDFS:

| Caminho no HDFS | Conteúdo |
|---|---|
| `/modelos/regressao_tarifa` | `PipelineModel` com a preparação e o melhor regressor |
| `/resultados/regressao_metricas` | CSV com as métricas de cada modelo |
| `/resultados/regressao_predicoes` | Predições no conjunto de teste |

## Conferindo os resultados

```bash
docker exec namenode hdfs dfs -cat /resultados/regressao_metricas/part-*.csv
docker exec namenode hdfs dfs -ls -R /modelos/regressao_tarifa
```

Para reaproveitar o modelo salvo no shell do PySpark (`docker exec -it spark-master /spark/bin/pyspark --master spark://spark-master:7077`):

```python
from pyspark.ml import PipelineModel
modelo = PipelineModel.load("hdfs://namenode:9000/modelos/regressao_tarifa")
```

O modelo espera as colunas `trip_distance`, `trip_duration_min`, `passenger_count`, `pickup_hour`, `eh_fim_de_semana` e `payment_label` (`"credito"` ou `"dinheiro"`).

## Observações

- **numpy:** as imagens `bde2020` (Alpine 3.10, Python 3.7) não trazem o numpy, que o `pyspark.ml` exige. O `setup.sh` instala o pacote via `apk`. Se um container for recriado, rode o `setup.sh` de novo.
- **Versões:** o notebook foi escrito para o Spark 4.0 (Colab). O script usa só APIs disponíveis no Spark 3.3 e sintaxe compatível com Python 3.7.
- **Tempo de treino:** o GBT com 50 iterações é o modelo mais lento. Em um Codespace de 2 núcleos, reduza `maxIter` ou `numTrees` se o treino demorar demais.

## Encerrando

```bash
cd docker-hadoop && docker-compose stop && cd ..
cd docker-spark && docker-compose stop && cd ..
```

Para reiniciar do zero, incluindo os dados do HDFS, use `docker-compose down -v` nas duas pastas.
