#!/usr/bin/env bash
#
# setup.sh — Sobe o cluster Hadoop + Spark (bde2020) no Codespaces, sempre do
# jeito certo:
#   1. Corrige a política da chain FORWARD do iptables quando estiver em DROP
#      (comum em ambientes Docker-in-Docker como o Codespaces, e que impede
#      containers de se conectarem entre si mesmo estando na mesma rede).
#      A correção é mínima por design: ajusta só a política, sem trocar o
#      backend padrão do iptables do sistema nem reiniciar o Docker -- versões
#      recentes do Docker gerenciam cadeias próprias (DOCKER-FORWARD) que
#      dependem especificamente do backend nft, e uma correção mais agressiva
#      (trocar para legacy + reiniciar o Docker) pode quebrar essa gestão
#      interna em alguns ambientes.
#   2. Cria a rede compartilhada ANTES do docker-compose up, evitando o erro
#      "UnknownHostException: namenode" entre os dois projetos separados.
#   3. Instala o numpy nos containers Spark (exigido pelo pyspark.ml).
#   4. Baixa do site da TLC os arquivos mensais (mês de treino + meses da
#      predição em batch) e os envia ao HDFS, se ainda não estiverem lá.
#   5. Copia os scripts Python para /app no spark-master.
#   6. Gera no HDFS as amostras: 200.000 linhas do mês de treino e 10.000
#      de cada mês da predição (job Spark gerar_amostras.py).
#
# Uso:
#   chmod +x setup.sh
#   ./setup.sh
#
# Para usar outros meses, defina as variáveis na chamada. Os scripts de treino
# e predição descobrem sozinhos as amostras geradas no HDFS:
#   MES_TREINO=3 MESES_PREDICAO="4 8 11" ./setup.sh
#
# Pode rodar mais de uma vez sem problema (é idempotente): se os repositórios
# já estiverem clonados, não clona de novo; se a rede já existir, não recria;
# se os containers já estiverem de pé, o `docker-compose up -d` só garante
# que continuam no ar; se a rede do SO já estiver correta, não mexe em nada.

set -uo pipefail

REDE="hadoop_spark_net"
SCRIPTS_SPARK="preparacao.py gerar_amostras.py treino_regressao.py predicao_batch.py"
CONTAINERS_SPARK="spark-master spark-worker-1 spark-worker-2"
ANO="2024"
MES_TREINO="${MES_TREINO:-1}"
MESES_PREDICAO="${MESES_PREDICAO:-2 6 12}"
N_TREINO="${N_TREINO:-200000}"        # linhas da amostra de treino
N_PREDICAO="${N_PREDICAO:-10000}"     # linhas da amostra de cada mês de predição
URL_TLC="https://d37ci6vzurychx.cloudfront.net/trip-data"
DIR_HDFS_TLC="/dados/tlc"

# ------------------------------------------------------------------------
# 1/9 — Corrigir problemas de rede do SO (ip_forward / iptables)
# ------------------------------------------------------------------------
echo "==> 1/9 Verificando compatibilidade de rede do host (ip_forward / iptables)..."

if [ "$(sysctl -n net.ipv4.ip_forward 2>/dev/null)" != "1" ]; then
  echo "    net.ipv4.ip_forward estava desligado -- ligando."
  sudo sysctl -w net.ipv4.ip_forward=1 > /dev/null
else
  echo "    net.ipv4.ip_forward já está ligado."
fi

PRECISA_AVISAR_RESET_MANUAL=false

# Confere a política da chain FORWARD nos dois conjuntos de regras possíveis
# (iptables "moderno"/nft e iptables-legacy). Em ambientes Docker-in-Docker
# como o Codespaces, os dois podem coexistir -- e o kernel pode aplicar um
# conjunto diferente do que `iptables` (sem sufixo) está mostrando, fazendo
# com que uma política DROP passe despercebida numa checagem simples.
#
# IMPORTANTE: corrigimos a política (-P FORWARD ACCEPT) diretamente em cada
# conjunto, SEM trocar qual conjunto é o "padrão do sistema"
# (update-alternatives) e SEM reiniciar o Docker. Versões recentes do Docker
# gerenciam cadeias próprias (ex: DOCKER-FORWARD) esperando especificamente o
# backend nft -- forçar o sistema inteiro para o legacy quebra essa gestão
# interna do próprio Docker em alguns ambientes (erro visto: "No chain/target
# /match by that name" ao tentar recriar DOCKER-FORWARD). Ajustar só a
# política, sem trocar o binário padrão, é a correção mínima e segura.
for BIN in iptables iptables-legacy; do
  if command -v "$BIN" > /dev/null 2>&1; then
    POLICY=$(sudo "$BIN" -L FORWARD -n 2>/dev/null | head -1 | sed -n 's/.*(policy \([A-Za-z]*\).*/\1/p')
    if [ "$POLICY" = "DROP" ] || [ "$POLICY" = "REJECT" ]; then
      echo "    $BIN: chain FORWARD com política $POLICY -- isso bloqueia tráfego"
      echo "    entre containers em redes diferentes. Corrigindo para ACCEPT..."
      sudo "$BIN" -P FORWARD ACCEPT
      PRECISA_AVISAR_RESET_MANUAL=true
    elif [ -n "$POLICY" ]; then
      echo "    $BIN: chain FORWARD já está com política $POLICY, ok."
    fi
  fi
done

if [ "$PRECISA_AVISAR_RESET_MANUAL" = true ]; then
  echo "    Política(s) corrigida(s). Isso normalmente já é suficiente -- o"
  echo "    script NÃO reinicia o Docker nem troca o backend padrão de"
  echo "    iptables, pois isso pode quebrar a gestão interna de rede de"
  echo "    versões recentes do Docker. Se, mesmo assim, os containers não"
  echo "    se conectarem entre si mais adiante, rode 'sudo service docker"
  echo "    restart' manualmente e tente de novo."
fi

# ------------------------------------------------------------------------
# 2/9 — Verificar o Docker
# ------------------------------------------------------------------------
echo "==> 2/9 Verificando o Docker..."
if ! command -v docker > /dev/null 2>&1; then
  echo "ERRO: o comando 'docker' não existe neste ambiente. Isso indica que o"
  echo "Codespace foi criado sem a feature 'docker-in-docker' (ex: antes do"
  echo ".devcontainer/devcontainer.json existir no repositório). Reconstrua o"
  echo "container: Ctrl+Shift+P -> 'Codespaces: Rebuild Container'."
  exit 1
fi

# Logo após o Codespace abrir, o dockerd pode ainda estar subindo -- ou pode
# ter falhado no boot. Espera um pouco e, se não responder, tenta iniciá-lo
# pelo entrypoint da feature docker-in-docker.
if ! docker info > /dev/null 2>&1; then
  echo "    Daemon do Docker ainda não respondeu -- aguardando até 30s..."
  for _ in $(seq 1 15); do
    docker info > /dev/null 2>&1 && break
    sleep 2
  done
fi

if ! docker info > /dev/null 2>&1; then
  echo "    Tentando iniciar o daemon do Docker manualmente..."
  if [ -x /usr/local/share/docker-init.sh ]; then
    sudo /usr/local/share/docker-init.sh > /tmp/docker-init.log 2>&1 &
  else
    sudo service docker start > /tmp/docker-init.log 2>&1
  fi
  for _ in $(seq 1 15); do
    docker info > /dev/null 2>&1 && break
    sleep 2
  done
fi

if ! docker info > /dev/null 2>&1; then
  echo "ERRO: o daemon do Docker não respondeu. Mensagem do 'docker info':"
  docker info 2>&1 | tail -5 | sed 's/^/      /'
  echo "Log da tentativa de inicialização: /tmp/docker-init.log"
  echo "Se persistir, reconstrua o container:"
  echo "Ctrl+Shift+P -> 'Codespaces: Rebuild Container'."
  exit 1
fi
echo "    Docker respondendo, ok."

# ------------------------------------------------------------------------
# 3/9 — Rede compartilhada entre os dois docker-compose
# ------------------------------------------------------------------------
echo "==> 3/9 Criando a rede compartilhada '$REDE' (se ainda não existir)..."
if docker network inspect "$REDE" > /dev/null 2>&1; then
  echo "    Rede '$REDE' já existe, seguindo."
else
  TENTATIVAS=0
  until docker network create "$REDE" > /dev/null 2>&1; do
    TENTATIVAS=$((TENTATIVAS + 1))
    if [ "$TENTATIVAS" -ge 5 ]; then
      echo "ERRO: não consegui criar a rede '$REDE' depois de $TENTATIVAS tentativas."
      echo "Rode 'docker network create $REDE' manualmente para ver a mensagem de erro exata."
      exit 1
    fi
    echo "    Falha ao criar a rede (tentativa $TENTATIVAS/5) -- tentando de novo em 2s..."
    sleep 2
  done
  echo "    Rede '$REDE' criada."
fi

# Confirma de verdade que a rede existe antes de seguir -- não custa nada e
# evita prosseguir com uma suposição incorreta.
if ! docker network inspect "$REDE" > /dev/null 2>&1; then
  echo "ERRO: a rede '$REDE' deveria existir agora, mas 'docker network inspect' não a encontrou."
  echo "Rode 'docker network ls' para investigar antes de continuar."
  exit 1
fi

# ------------------------------------------------------------------------
# 4/9 — Clonar os repositórios (se necessário) e aplicar o override de rede
# ------------------------------------------------------------------------
echo "==> 4/9 Clonando docker-hadoop e docker-spark (se necessário)..."
for REPO in docker-hadoop docker-spark; do
  if [ -f "$REPO/docker-compose.yml" ]; then
    echo "    '$REPO' já está presente, seguindo."
  else
    echo "    '$REPO' ausente ou incompleto — (re)clonando..."
    rm -rf "$REPO"
    git clone --depth 1 "https://github.com/big-data-europe/$REPO.git"
  fi

  # Garante que o override de rede está presente, mesmo em clones antigos
  cat > "$REPO/docker-compose.override.yml" << EOF
networks:
  default:
    name: $REDE
    external: true
EOF
done

# A branch atual do docker-spark define só 1 worker (spark-worker-1) por padrão.
# Para atividades que precisam de mais de um worker (ex: demonstrar tolerância
# a falhas matando um worker no meio de um job), adicionamos um segundo aqui,
# reaproveitando a mesma imagem e variáveis de ambiente do primeiro.
if ! grep -q "spark-worker-2" docker-spark/docker-compose.override.yml 2>/dev/null; then
  IMAGEM_WORKER=$(grep -A1 "^  spark-worker-1:" docker-spark/docker-compose.yml | grep "image:" | awk '{print $2}')
  cat >> docker-spark/docker-compose.override.yml << EOF
services:
  spark-master:
    ports:
      - "4040:4040"
  spark-worker-2:
    image: ${IMAGEM_WORKER}
    container_name: spark-worker-2
    depends_on:
      - spark-master
    ports:
      - "8082:8081"
    environment:
      - SPARK_MASTER=spark://spark-master:7077
EOF
fi

# Ajuste preventivo no HDFS: sem isso, leituras e escritas podem falhar com
# "UnresolvedAddressException", porque os datanodes, nesse ambiente Docker,
# às vezes se registram usando um hostname interno bruto que não é resolvível
# pela rede. Aplicado ANTES do primeiro `docker-compose up` do docker-hadoop,
# pra ninguém bater nesse erro durante o experimento em aula.
if ! grep -q "dfs_client_use_datanode_hostname" docker-hadoop/hadoop.env 2>/dev/null; then
  cat >> docker-hadoop/hadoop.env << 'EOF'
HDFS_CONF_dfs_client_use_datanode_hostname=false
HDFS_CONF_dfs_datanode_use_datanode_hostname=false
EOF
fi

# ------------------------------------------------------------------------
# 5/9 — Subir os clusters e validar
# ------------------------------------------------------------------------
echo "==> 5/9 Subindo os clusters..."
( cd docker-hadoop && docker-compose up -d )
if [ $? -ne 0 ]; then
  echo "ERRO: falha ao subir o docker-hadoop (veja a mensagem acima)."
  echo "Confira 'docker network ls' -- a rede '$REDE' precisa existir antes deste passo."
  exit 1
fi

( cd docker-spark && docker-compose up -d )
if [ $? -ne 0 ]; then
  echo "ERRO: falha ao subir o docker-spark (veja a mensagem acima)."
  echo "Confira 'docker network ls' -- a rede '$REDE' precisa existir antes deste passo."
  exit 1
fi

echo "    Aguardando o namenode responder..."
TENTATIVAS=0
until docker exec namenode hdfs dfs -ls / > /dev/null 2>&1; do
  TENTATIVAS=$((TENTATIVAS + 1))
  if [ "$TENTATIVAS" -ge 20 ]; then
    echo "AVISO: o namenode não respondeu depois de várias tentativas."
    echo "Rode 'docker logs namenode' para investigar antes de continuar."
    break
  fi
  sleep 3
done

echo "    Verificando se há pelo menos um datanode vivo..."
TENTATIVAS=0
DATANODES_VIVOS=0
until [ "$DATANODES_VIVOS" -ge 1 ]; do
  DATANODES_VIVOS=$(docker exec namenode hdfs dfsadmin -report 2>/dev/null | sed -n 's/^Live datanodes (\([0-9]*\)).*/\1/p')
  DATANODES_VIVOS=${DATANODES_VIVOS:-0}
  if [ "$DATANODES_VIVOS" -ge 1 ]; then
    break
  fi
  TENTATIVAS=$((TENTATIVAS + 1))
  if [ "$TENTATIVAS" -ge 15 ]; then
    echo "AVISO: nenhum datanode vivo depois de várias tentativas."
    echo "Isso costuma acontecer quando o volume do namenode persistiu de uma"
    echo "sessão anterior mas o do datanode foi perdido/recriado (metadado"
    echo "aponta para blocos que não existem fisicamente mais -- erro típico:"
    echo "'BlockMissingException ... No live nodes contain current block')."
    echo "Se isso acontecer ao ler um arquivo do HDFS mais tarde, resolva com:"
    echo "  cd docker-hadoop && docker-compose down -v && docker-compose up -d && cd .."
    echo "e reenvie os dados pro HDFS depois."
    break
  fi
  sleep 3
done
if [ "$DATANODES_VIVOS" -ge 1 ]; then
  echo "    OK: $DATANODES_VIVOS datanode(s) vivo(s)."
fi

echo "    Verificando se Spark enxerga o Hadoop pela rede..."
if docker exec spark-master getent hosts namenode > /dev/null 2>&1; then
  echo "    OK: spark-master resolve 'namenode' normalmente."
else
  echo "AVISO: spark-master ainda não resolve 'namenode'. Rode:"
  echo "  docker exec -it spark-master getent hosts namenode"
  echo "para investigar (pode ser só questão de mais alguns segundos de boot)."
fi

echo "    Aguardando ao menos um worker Spark se conectar ao master..."
TENTATIVAS=0
until docker exec spark-master curl -s http://localhost:8080 2>/dev/null | grep -q "Alive Workers: [1-9]"; do
  TENTATIVAS=$((TENTATIVAS + 1))
  if [ "$TENTATIVAS" -ge 15 ]; then
    echo "AVISO: nenhum worker apareceu como 'Alive' na UI do Spark Master ainda."
    echo "Se isso persistir, confira 'docker logs spark-worker-1' -- um sintoma"
    echo "comum é 'Retrying connection to master' em loop, que geralmente é o"
    echo "mesmo problema de iptables corrigido no passo 1/9 (raro acontecer de"
    echo "novo depois da correção, mas pode acontecer em outro Codespace)."
    break
  fi
  sleep 4
done
if [ "$TENTATIVAS" -lt 15 ]; then
  echo "    OK: pelo menos um worker Spark está conectado."
fi

# ------------------------------------------------------------------------
# 6/9 — Instalar o numpy nos containers Spark
# ------------------------------------------------------------------------
# As imagens bde2020 (Alpine 3.10, Python 3.7) não trazem o numpy, mas o
# pyspark.ml depende dele. Instalamos pelo apk (pacote já compilado; via pip
# no Alpine seria uma compilação demorada). A instalação se perde se o
# container for recriado, por isso este passo roda sempre.
echo "==> 6/9 Instalando numpy nos containers Spark (necessário para o Spark ML)..."
for C in $CONTAINERS_SPARK; do
  if docker exec "$C" python3 -c "import numpy" > /dev/null 2>&1; then
    echo "    $C: numpy já instalado."
  elif docker exec "$C" apk add --no-cache py3-numpy > /dev/null 2>&1; then
    echo "    $C: numpy instalado."
  else
    echo "AVISO: falha ao instalar numpy em $C. Rode para ver o erro:"
    echo "  docker exec $C apk add --no-cache py3-numpy"
  fi
done

# ------------------------------------------------------------------------
# 7/9 — Arquivos mensais da TLC no HDFS
# ------------------------------------------------------------------------
echo "==> 7/9 Preparando os arquivos mensais da TLC no HDFS ($ANO)..."
docker exec namenode hdfs dfs -mkdir -p "$DIR_HDFS_TLC"
mkdir -p dados
for MES in $MES_TREINO $MESES_PREDICAO; do
  ARQUIVO=$(printf "yellow_tripdata_%s-%02d.parquet" "$ANO" "$((10#$MES))")
  if docker exec namenode hdfs dfs -test -e "$DIR_HDFS_TLC/$ARQUIVO" 2>/dev/null; then
    echo "    $ARQUIVO já está no HDFS, seguindo."
    continue
  fi
  if [ ! -s "dados/$ARQUIVO" ]; then
    echo "    Baixando $ARQUIVO..."
    if ! wget -q -O "dados/$ARQUIVO" "$URL_TLC/$ARQUIVO"; then
      echo "ERRO: falha ao baixar $URL_TLC/$ARQUIVO"
      rm -f "dados/$ARQUIVO"
      exit 1
    fi
  fi
  docker cp "dados/$ARQUIVO" "namenode:/tmp/$ARQUIVO"
  docker exec namenode hdfs dfs -put -f "/tmp/$ARQUIVO" "$DIR_HDFS_TLC/"
  docker exec namenode rm -f "/tmp/$ARQUIVO"
  echo "    $ARQUIVO enviado para $DIR_HDFS_TLC."
done

# ------------------------------------------------------------------------
# 8/9 — Copiar os scripts para o spark-master
# ------------------------------------------------------------------------
echo "==> 8/9 Copiando os scripts para /app no spark-master..."
docker exec spark-master mkdir -p /app
for S in $SCRIPTS_SPARK; do
  docker cp "$S" "spark-master:/app/$S"
done

# ------------------------------------------------------------------------
# 9/9 — Gerar as amostras de treino e de predição (job Spark no cluster)
# ------------------------------------------------------------------------
echo "==> 9/9 Gerando amostras: $N_TREINO linhas do mês $MES_TREINO (treino) e"
echo "    $N_PREDICAO linhas de cada mês em '$MESES_PREDICAO' (predição)..."
FALTA_AMOSTRA=false
for PAR in "treino:$MES_TREINO" $(for M in $MESES_PREDICAO; do echo "predicao:$M"; done); do
  AMOSTRA=$(printf "/dados/amostras/%s_%s-%02d" "${PAR%%:*}" "$ANO" "$((10#${PAR##*:}))")
  if ! docker exec namenode hdfs dfs -test -e "$AMOSTRA/_SUCCESS" 2>/dev/null; then
    FALTA_AMOSTRA=true
  fi
done

if [ "$FALTA_AMOSTRA" = false ] && [ "${REGERAR_AMOSTRAS:-0}" != "1" ]; then
  echo "    Amostras já existem no HDFS, seguindo."
  echo "    (para gerar de novo, ex. após mudar o tamanho: REGERAR_AMOSTRAS=1 ./setup.sh)"
elif ! docker exec spark-master /spark/bin/spark-submit \
       --master spark://spark-master:7077 /app/gerar_amostras.py \
       --mes-treino "$MES_TREINO" --n-treino "$N_TREINO" \
       --meses $MESES_PREDICAO --n-predicao "$N_PREDICAO" 2>&1 \
       | grep -E "^(treino|predicao) |Error|Exception"; then
  echo "AVISO: não foi possível confirmar a geração das amostras. Rode o job"
  echo "manualmente para ver a saída completa:"
  echo "  docker exec -it spark-master /spark/bin/spark-submit --master spark://spark-master:7077 \\"
  echo "    /app/gerar_amostras.py --mes-treino $MES_TREINO --n-treino $N_TREINO \\"
  echo "    --meses $MESES_PREDICAO --n-predicao $N_PREDICAO"
fi

SUBMIT="docker exec -it spark-master /spark/bin/spark-submit --master spark://spark-master:7077"
echo ""
echo "==> Cluster pronto. Próximos passos:"
echo "    1) Treinar com o mês $MES_TREINO:"
echo "       $SUBMIT /app/treino_regressao.py"
echo "    2) Predição em batch nos meses $MESES_PREDICAO:"
echo "       $SUBMIT /app/predicao_batch.py"
echo "    3) Trazer o CSV de predições para o Codespace:"
echo "       ./exportar_resultados.sh"
echo ""
echo "    Interfaces web (aba 'Ports' do Codespace):"
echo "    Namenode (HDFS)         -> porta 9870"
echo "    Spark Master            -> porta 8080"
echo "    Spark Application UI    -> porta 4040 (só durante a execução)"
