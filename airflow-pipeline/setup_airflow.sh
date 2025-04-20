#!/usr/bin/env bash

# Bootstrap Airflow pipeline:
# 1) Copy Graphrag outputs
# 2) Generate .env
# 3) Init DB & create user
# 4) Start webserver, scheduler, worker

set -euo pipefail

# Absolute path to this script's directory (airflow-pipeline)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths
DATA_SRC="${DIR}/../graphrag_outputs"
DATA_DST="${DIR}/data/graphrag_outputs"
ENV_FILE="${DIR}/.env"

echo "1) Copiando dados Parquet: ${DATA_SRC} -> ${DATA_DST}"
mkdir -p "${DIR}/data"
rm -rf "${DATA_DST}"
cp -r "${DATA_SRC}" "${DATA_DST}"

echo "2) Gerando arquivo .env em ${ENV_FILE}"
cat > "${ENV_FILE}" <<EOF
OPENAI_API_KEY=${OPENAI_API_KEY:-}
AIRFLOW_UID=$(id -u)
AIRFLOW_GID=$(id -g)
EOF

echo "3) Inicializando banco e criando usuário Admin"
cd "${DIR}"
docker-compose up -d airflow-init
docker-compose logs airflow-init --tail=50

echo "4) Subindo Airflow: Webserver, Scheduler e Worker"
docker-compose up -d airflow-webserver airflow-scheduler airflow-worker

echo
echo "Airflow iniciado! Acesse: http://localhost:8080 (user/admin, pwd/admin)"