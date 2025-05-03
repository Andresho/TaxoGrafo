# Use a imagem oficial do Airflow como base
# Certifique-se que a versão é compatível com suas necessidades
FROM apache/airflow:2.8.4-python3.10

# Argumento para definir o usuário Airflow (use seu UID local)
ARG AIRFLOW_UID=50000
ARG AIRFLOW_GID=0

# Define variáveis de ambiente do Airflow
ENV AIRFLOW_HOME=/opt/airflow
# Adiciona o diretório local do usuário ao PYTHONPATH e PATH explicitamente
# para garantir visibilidade dos pacotes instalados com --user
ENV PYTHONPATH="${AIRFLOW_HOME}/.local/lib/python3.9/site-packages:${PYTHONPATH}"
ENV PATH="${AIRFLOW_HOME}/.local/bin:${PATH}"

# Copia requirements.txt PRIMEIRO para aproveitar o cache do Docker
# Garante que pertença ao usuário correto desde o início
COPY --chown=${AIRFLOW_UID}:${AIRFLOW_GID} requirements.txt /requirements.txt

# Mude para o usuário airflow ANTES de instalar com --user
USER $AIRFLOW_UID

# Instala as dependências Python como usuário airflow
# O aviso sobre --user pode aparecer, mas a instalação deve ocorrer
# Remova 'apache-airflow' deste arquivo se estiver presente
RUN python -m pip install --no-cache-dir --user -r /requirements.txt

# Mudar temporariamente para root para criar/ajustar diretórios base
USER root
RUN mkdir -p ${AIRFLOW_HOME}/dags ${AIRFLOW_HOME}/logs ${AIRFLOW_HOME}/plugins ${AIRFLOW_HOME}/scripts ${AIRFLOW_HOME}/data \
    && chown -R ${AIRFLOW_UID}:${AIRFLOW_GID} ${AIRFLOW_HOME}


COPY --chown=${AIRFLOW_UID}:${AIRFLOW_GID} dags/ /opt/airflow/dags/
## Removido setup do Graphrag (agora isolado em container específico)
## RUN graphrag init, prompts e settings não são mais necessários aqui

# Switch back to the Airflow user for normal execution
USER ${AIRFLOW_UID}

# Comando padrão da imagem (geralmente `airflow standalone` ou similar,
# mas será sobrescrito pelo docker-compose para a maioria dos serviços)
# HEALTHCHECK pode ser adicionado aqui se necessário, mas já está no compose

WORKDIR ${AIRFLOW_HOME}