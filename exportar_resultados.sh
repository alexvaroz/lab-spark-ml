#!/usr/bin/env bash
#
# exportar_resultados.sh — Copia os CSVs gerados pelo Spark no HDFS para a
# pasta ./saida do Codespace. De lá, é só clicar com o botão direito no
# arquivo (Explorer do VS Code) -> Download para baixar no seu computador.
#
# O Spark grava cada resultado como uma PASTA (com um arquivo part-*.csv
# dentro); aqui cada pasta vira um único arquivo .csv com nome legível.

set -uo pipefail

mkdir -p saida

for RESULTADO in predicoes_batch metricas_por_mes regressao_metricas; do
  ORIGEM="/resultados/$RESULTADO"
  if ! docker exec namenode hdfs dfs -test -e "$ORIGEM" 2>/dev/null; then
    echo "    $ORIGEM não existe no HDFS (o job correspondente já rodou?) -- pulando."
    continue
  fi
  docker exec namenode hdfs dfs -cat "$ORIGEM/part-*.csv" > "saida/$RESULTADO.csv"
  echo "    $ORIGEM -> saida/$RESULTADO.csv ($(($(wc -l < "saida/$RESULTADO.csv") - 1)) linhas)"
done
