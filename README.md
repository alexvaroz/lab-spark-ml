# Laboratório — Regressão com Spark ML em cluster Hadoop + Spark

Treina um modelo de regressão que prevê a tarifa (`fare_amount`) das corridas de táxi de Nova York com **Spark ML**. O treino usa um mês de 2024, e o modelo depois faz a **predição em batch** de amostras de outros meses. Tudo roda em um cluster real Hadoop (HDFS) + Spark Standalone, em containers no **GitHub Codespaces**.

É a continuação do laboratório de processamento distribuído. Usa as mesmas imagens `bde2020` (Spark 3.3.0, Hadoop 3) e a seção de regressão do notebook `unidade_5.ipynb`.

## Estrutura

| Arquivo | Função |
|---|---|
| `.devcontainer/devcontainer.json` | Codespace Ubuntu com Docker-in-Docker |
| `setup.sh` | Sobe o cluster, instala o numpy, baixa os meses da TLC, envia ao HDFS e gera as amostras |
| `preparacao.py` | Leitura, limpeza e atributos, usados igualmente pelo treino e pela predição |
| `gerar_amostras.py` | Gera as amostras de treino e de predição no HDFS (chamado pelo `setup.sh`) |
| `treino_regressao.py` | **Etapa 1:** treina 3 regressores e salva no HDFS os pipelines e os metadados |
| `predicao_batch.py` | **Etapa 2:** carrega um pipeline salvo e faz a predição nas amostras dos outros meses |
| `exportar_resultados.sh` | Copia os CSVs de resultado do HDFS para a pasta `saida/` do Codespace |
| `modelos.sh` | Exporta os modelos do HDFS para `modelos_salvos/` e os importa de volta |
| `experimento_regressao_colab.ipynb` | O mesmo experimento completo em um notebook para o Google Colab, sem cluster: Spark local e disco no lugar do HDFS |

## Fluxo

```
Site da TLC ──wget──► HDFS /dados/tlc/yellow_tripdata_2024-MM.parquet   (mês completo)
                         │  gerar_amostras.py (limpeza + amostragem)
                         ▼
             HDFS /dados/amostras/treino_2024-01      (200.000 linhas)
             HDFS /dados/amostras/predicao_2024-MM    (10.000 linhas por mês)
                         │
     Etapa 1: treino_regressao.py ─► /modelos  (pipelines + metadados)
                         │                  ▲
                         │                  └─ modelos.sh exportar/importar ◄─► modelos_salvos/
                         ▼
     Etapa 2: predicao_batch.py ─► /resultados/predicoes_batch + /resultados/metricas_por_mes
                         │  exportar_resultados.sh
                         ▼
                   saida/*.csv no Codespace ──► Download
```

## Executando

1. Abra o repositório em um Codespace (**Code → Codespaces → Create codespace**).
2. Prepare o ambiente:

   ```bash
   chmod +x setup.sh exportar_resultados.sh modelos.sh
   ./setup.sh
   ```

   Por padrão, o treino usa **janeiro** e a predição usa **fevereiro, junho e dezembro**. Para escolher outros meses ou outros tamanhos de amostra:

   ```bash
   MES_TREINO=3 MESES_PREDICAO="4 8 11" N_TREINO=200000 N_PREDICAO=10000 ./setup.sh
   ```

   Não é preciso repetir os meses nos passos 3 e 4: os scripts descobrem sozinhos as amostras que existem no HDFS. As amostras que já existem não são refeitas. Para gerá-las de novo, por exemplo depois de mudar o tamanho, rode `REGERAR_AMOSTRAS=1 ./setup.sh`.

3. **Etapa 1 — treino.** Treina com a amostra do mês de treino e salva os pipelines no HDFS:

   ```bash
   docker exec -it spark-master /spark/bin/spark-submit \
       --master spark://spark-master:7077 /app/treino_regressao.py
   ```

   Se houver amostras de treino de mais de um mês, escolha uma com `--mes 3`, por exemplo.

4. **Etapa 2 — predição em batch.** Não treina nada: carrega o pipeline salvo e o aplica aos outros meses.

   ```bash
   docker exec -it spark-master /spark/bin/spark-submit \
       --master spark://spark-master:7077 /app/predicao_batch.py
   ```

   Sem `--meses`, a predição usa **todos** os meses com amostra de predição no HDFS, inclusive os de execuções anteriores do `setup.sh`. Para restringir, passe por exemplo `--meses 4`. Por padrão, usa o melhor modelo do treino. Para usar outro, passe `--modelo linearregression`, `--modelo randomforest` ou `--modelo gbt`.

5. Traga os resultados para fora do HDFS:

   ```bash
   ./exportar_resultados.sh
   ```

   Os CSVs vão para a pasta `saida/`. Para baixá-los no seu computador, clique com o botão direito no arquivo, no Explorer do VS Code, e escolha **Download**.

Acompanhe os jobs nas portas **8080** (Spark Master) e **4040** (estágios do job, só durante a execução).

## Resultados

| Arquivo em `saida/` | Origem no HDFS | Conteúdo |
|---|---|---|
| `predicoes_batch.csv` | `/resultados/predicoes_batch` | Uma linha por corrida: `mes`, data e hora do embarque, atributos, `fare_amount` (real), `tarifa_prevista` e `erro_abs` |
| `metricas_por_mes.csv` | `/resultados/metricas_por_mes` | RMSE, MAE e R² do modelo em cada mês da predição |
| `regressao_metricas.csv` | `/resultados/regressao_metricas` | Métricas dos 3 regressores no conjunto de teste do mês de treino |

Comparar `metricas_por_mes.csv` com `regressao_metricas.csv` mostra se o modelo treinado em um mês **generaliza** para os outros. A tarifa de NYC muda pouco ao longo do ano, mas sazonalidade e trânsito podem pesar.

## Modelos e pipelines salvos

A etapa 1 salva em `/modelos` no HDFS:

| Caminho | Conteúdo |
|---|---|
| `/modelos/preparacao` | Pipeline de preparação ajustado no treino: Imputer → StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler → VectorAssembler |
| `/modelos/pipeline_linearregression` | Preparação + LinearRegression |
| `/modelos/pipeline_randomforest` | Preparação + RandomForestRegressor |
| `/modelos/pipeline_gbt` | Preparação + GBTRegressor |
| `/modelos/melhor` | Cópia do pipeline completo de menor RMSE |
| `/modelos/metadados` | JSON com o mês de treino, o melhor modelo, as métricas de cada um e as colunas de entrada |

Os pipelines completos (`PipelineModel`) recebem as colunas `trip_distance`, `trip_duration_min`, `passenger_count`, `pickup_hour`, `eh_fim_de_semana` e `payment_label` (`"credito"` ou `"dinheiro"`). Eles devolvem a coluna `prediction`. Para usá-los em outro código:

```python
from pyspark.ml import PipelineModel
modelo = PipelineModel.load("hdfs://namenode:9000/modelos/melhor")
```

**Guardando os modelos fora do HDFS.** O HDFS é apagado com `docker-compose down -v` ou quando o Codespace é recriado. Para rodar a etapa 2 em outra sessão sem treinar de novo:

```bash
./modelos.sh exportar   # HDFS /modelos -> ./modelos_salvos (dá para fazer commit dessa pasta)
# ... em outra sessão, depois do ./setup.sh:
./modelos.sh importar   # ./modelos_salvos -> HDFS /modelos
```

## Observações

- **Dados:** os arquivos mensais oficiais da [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) têm de 50 a 60 MB e cerca de 3 milhões de corridas cada. As amostras são tiradas **depois** da limpeza, então todas as linhas são corridas válidas. Também são descartados os registros com data fora do mês do arquivo, que os arquivos oficiais trazem.
- **numpy:** as imagens `bde2020` (Alpine 3.10, Python 3.7) não trazem o numpy, que o `pyspark.ml` exige. O `setup.sh` instala o pacote via `apk`. Se um container for recriado, rode o `setup.sh` de novo.
- **Versões:** o notebook foi escrito para o Spark 4.0 (Colab). Os scripts usam só APIs disponíveis no Spark 3.3 e sintaxe compatível com Python 3.7.

## Encerrando

```bash
cd docker-hadoop && docker-compose stop && cd ..
cd docker-spark && docker-compose stop && cd ..
```

Para reiniciar do zero, incluindo os dados do HDFS, use `docker-compose down -v` nas duas pastas.
