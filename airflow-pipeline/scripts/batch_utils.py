import logging
import json
import uuid
import pandas as pd
import scripts.pipeline_tasks as pt

def check_batch_status(batch_id: str):
    """Consulta status do batch e retorna (status, output_file_id, error_file_id)."""
    if not pt.OPENAI_CLIENT:
        raise ValueError("OpenAI client não inicializado")
    try:
        batch_job = pt.OPENAI_CLIENT.batches.retrieve(batch_id)
        logging.info(f"Status do Batch {batch_id}: {batch_job.status}")
        return batch_job.status, batch_job.output_file_id, batch_job.error_file_id
    except Exception as e:
        logging.error(f"Erro ao verificar status do batch {batch_id}: {e}")
        return "API_ERROR", None, None

def process_batch_results(
    batch_id: str,
    output_file_id: str,
    error_file_id: str,
    stage_output_dir,
    output_filename: str
) -> bool:
    """Baixa e processa o arquivo de resultados do batch."""
    if not pt.OPENAI_CLIENT:
        raise ValueError("OpenAI client não inicializado")
    logging.info(f"Processando resultados do Batch {batch_id} (Output File: {output_file_id})...")
    processed_data = []
    errors_in_batch = 0
    all_ok = True
    try:
        # Logar erros do batch se existirem
        if error_file_id:
            try:
                error_content = pt.OPENAI_CLIENT.files.content(error_file_id).read().decode('utf-8')
                logging.warning(f"Erros individuais no batch {batch_id}:\n{error_content[:1000]}...")
            except Exception as ef:
                logging.error(f"Não leu arq erro {error_file_id}: {ef}")
        # Baixar e processar resultados
        result_content_bytes = pt.OPENAI_CLIENT.files.content(output_file_id).read()
        result_content = result_content_bytes.decode('utf-8')
        logging.info(f"Arquivo de resultado {output_file_id} baixado.")
        for line in result_content.strip().split('\n'):
            try:
                line_data = json.loads(line)
                custom_id = line_data.get("custom_id", "unknown_custom_id")
                if custom_id.startswith("gen_req_"):
                    origin_id = "_".join(custom_id.split("_")[2:-1])
                else:
                    origin_id = custom_id
                response = line_data.get("response")
                error = line_data.get("error")
                if error:
                    errors_in_batch += 1
                    logging.error(f"Erro batch {custom_id}: {error.get('message')}")
                    continue
                if not response or response.get("status_code") != 200:
                    logging.warning(f"Request {custom_id} status não OK: {response}")
                    errors_in_batch += 1
                    continue
                body = response.get("body", {})
                choice = (body.get("choices") or [{}])[0]
                message_content = choice.get("message", {}).get("content")
                if not message_content:
                    logging.warning(f"Resposta OK sem conteúdo {custom_id}")
                    errors_in_batch += 1
                    continue
                content_cleaned = message_content.strip()
                if content_cleaned.startswith("```json"):
                    content_cleaned = content_cleaned[7:-3].strip()
                elif content_cleaned.startswith("```"):
                    content_cleaned = content_cleaned[3:-3].strip()
                try:
                    inner_data = json.loads(content_cleaned)
                    if output_filename == pt.GENERATED_UCS_RAW:
                        units = inner_data.get("generated_units", [])
                        if isinstance(units, list) and len(units) == 6:
                            for unit in units:
                                if isinstance(unit, dict) and "bloom_level" in unit and "uc_text" in unit:
                                    unit["uc_id"] = str(uuid.uuid4())
                                    unit["origin_id"] = origin_id
                                    processed_data.append(unit)
                        else:
                            logging.warning(f"JSON interno {custom_id} != 6 UCs")
                            errors_in_batch += 1
                    elif output_filename == pt.UC_EVALUATIONS_RAW:
                        assessments = inner_data.get("difficulty_assessments", [])
                        if isinstance(assessments, list):
                            for assessment in assessments:
                                if isinstance(assessment, dict) and 'uc_id' in assessment and 'difficulty_score' in assessment:
                                    processed_data.append(assessment)
                                else:
                                    logging.warning(f"Formato assessment inválido para {custom_id}")
                                    errors_in_batch += 1
                        else:
                            logging.warning(f"JSON interno {custom_id} sem lista 'difficulty_assessments'")
                            errors_in_batch += 1
                except json.JSONDecodeError as e:
                    logging.error(f"Erro JSON decode interno {custom_id}: {e}")
                    errors_in_batch += 1
            except Exception as e:
                logging.error(f"Erro processando linha: {e}. Linha: {line[:100]}...")
                errors_in_batch += 1
        logging.info(f"Processamento arquivo concluído. {len(processed_data)} registros. {errors_in_batch} erros.")
        if processed_data:
            df = pd.DataFrame(processed_data)
            pt.save_dataframe(df, stage_output_dir, output_filename)
            if errors_in_batch > 0:
                logging.warning("Processamento concluído, mas com erros individuais no batch.")
            all_ok = True
        elif errors_in_batch > 0:
            logging.error("Nenhum dado processado com sucesso do batch.")
            all_ok = False
        else:
            logging.warning("Nenhum dado encontrado no arquivo de resultados do batch.")
            all_ok = True
    except Exception:
        logging.exception(f"Falha ao baixar/processar arquivo {output_file_id} do batch {batch_id}.")
        all_ok = False
    return all_ok