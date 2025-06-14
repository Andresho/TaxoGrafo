#!/bin/bash

# Este script MESCLA os dados da execução do TCC ('tcc_run')
# a um banco de dados existente, ignorando registros duplicados.

# --- Variáveis de Configuração ---
DB_USER="app_user"
DB_NAME="app_data"
POSTGRES_SERVICE_NAME="postgres"
MERGE_DUMP_FILE="./test_runs/tcc_run/tcc_run_dbdump.sql"
ARTIFACTS_DIR="./test_runs/tcc_run/tcc_run/"
TARGET_ARTIFACTS_DIR="./data/tcc_run/"

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Iniciando a MESCLAGEM dos dados da execução do TCC...${NC}"

# --- Verificações ---
echo "Verificando pré-requisitos..."
if ! command -v docker-compose &> /dev/null; then echo -e "${RED}Erro: docker-compose não encontrado.${NC}"; exit 1; fi
if [ ! -f "$MERGE_DUMP_FILE" ]; then echo -e "${RED}Erro: Arquivo de dump '$MERGE_DUMP_FILE' não encontrado.${NC}"; exit 1; fi
POSTGRES_CONTAINER_ID=$(docker-compose ps -q "$POSTGRES_SERVICE_NAME")
if [ -z "$POSTGRES_CONTAINER_ID" ]; then
    echo -e "${RED}Erro: O contêiner '$POSTGRES_SERVICE_NAME' não está em execução. Execute 'docker-compose up -d'.${NC}"
    exit 1
fi
echo "Pré-requisitos verificados."
echo ""

# --- Cópia dos Artefatos ---
echo -e "${GREEN}--> Passo 1: Copiando artefatos do GraphRAG...${NC}"
mkdir -p "$TARGET_ARTIFACTS_DIR"
rsync -av "$ARTIFACTS_DIR" "$TARGET_ARTIFACTS_DIR" # rsync sem --delete para não apagar outras runs
echo "Artefatos copiados."
echo ""

# --- Mesclagem no Banco de Dados ---
echo -e "${GREEN}--> Passo 2: Mesclando dados no banco de dados (ignorando duplicatas)...${NC}"
# Importa o dump "mesclável" que usa INSERT ... ON CONFLICT DO NOTHING
docker exec -i "$POSTGRES_CONTAINER_ID" psql -U "$DB_USER" -d "$DB_NAME" < "$MERGE_DUMP_FILE" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "Dados mesclados com sucesso."
else
    echo -e "${RED}Ocorreram erros durante a mesclagem. Verifique se o banco de dados e o usuário existem.${NC}"
fi
echo ""

# --- Conclusão ---
echo -e "${GREEN}✅ Mesclagem concluída!${NC}"
echo "Os dados da execução 'tcc_run' foram adicionados ao banco de dados."
echo "Explore os resultados em: http://localhost:8000/docs"