#!/usr/bin/env bash
#
# modelos.sh — Guarda os modelos e pipelines treinados fora do HDFS, e os
# devolve ao HDFS depois.
#
# O HDFS do cluster é apagado com `docker-compose down -v` ou quando o
# Codespace é recriado. Exportando para ./modelos_salvos (pasta comum do
# Codespace, que pode ir para o git ou ser baixada), dá para rodar a etapa de
# predição em outra sessão sem treinar de novo.
#
# Uso:
#   ./modelos.sh exportar    # HDFS /modelos  -> ./modelos_salvos
#   ./modelos.sh importar    # ./modelos_salvos -> HDFS /modelos

set -uo pipefail

LOCAL="modelos_salvos"
HDFS_DIR="/modelos"
TMP="/tmp/modelos_transferencia"

case "${1:-}" in
  exportar)
    if ! docker exec namenode hdfs dfs -test -e "$HDFS_DIR/metadados" 2>/dev/null; then
      echo "ERRO: $HDFS_DIR/metadados não existe no HDFS -- rode o treino antes."
      exit 1
    fi
    docker exec namenode rm -rf "$TMP"
    docker exec namenode hdfs dfs -get "$HDFS_DIR" "$TMP" || exit 1
    rm -rf "$LOCAL"
    docker cp "namenode:$TMP" "$LOCAL" || exit 1
    docker exec namenode rm -rf "$TMP"
    echo "Modelos exportados para ./$LOCAL:"
    ls -1 "$LOCAL"
    ;;
  importar)
    if [ ! -d "$LOCAL/metadados" ]; then
      echo "ERRO: ./$LOCAL não contém modelos exportados (falta $LOCAL/metadados)."
      exit 1
    fi
    docker exec namenode rm -rf "$TMP"
    docker cp "$LOCAL" "namenode:$TMP" || exit 1
    docker exec namenode hdfs dfs -rm -r -f "$HDFS_DIR" > /dev/null
    docker exec namenode hdfs dfs -put "$TMP" "$HDFS_DIR" || exit 1
    docker exec namenode rm -rf "$TMP"
    echo "Modelos importados para o HDFS em $HDFS_DIR:"
    docker exec namenode hdfs dfs -ls "$HDFS_DIR"
    ;;
  *)
    echo "Uso: $0 exportar|importar"
    exit 1
    ;;
esac
