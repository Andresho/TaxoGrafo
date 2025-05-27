tree > ./allfiles/tree.txt

cp ./airflow-pipeline/dags/knowledge_graph_pipeline_dag.py ./allfiles
cp ./airflow-pipeline/postgres-init/01-create-pipeline-db.sql ./allfiles/01-create-pipeline-db.txt
cp ./airflow-pipeline/requirements.txt ./allfiles/requirements.txt

cp ./app/alembic/versions/*.py ./allfiles/
cp ./app/alembic/env.py ./allfiles/env.py

cp ./app/crud/*.py ./allfiles/

cp ./app/scripts/llm_core/models.py ./allfiles/
cp ./app/scripts/llm_providers/openai_utils.py ./allfiles/
cp ./app/scripts/pipeline_stages/*.py ./allfiles/
cp ./app/scripts/* ./allfiles/
cp ./app/* ./allfiles/

cp ./dockerfiles/Dockerfile.airflow ./allfiles/Dockerfile_aiflow.txt
cp ./dockerfiles/Dockerfile.api ./allfiles/Dockerfile_api.txt
cp ./dockerfiles/Dockerfile.graphrag ./allfiles/Dockerfile_graphrag.txt