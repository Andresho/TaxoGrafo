import pandas as pd

# import getpass
# import os
# from typing import Annotated
# from typing_extensions import TypedDict
#
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langchain_openai import ChatOpenAI
#
# from pathlib import Path
#
# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = "..."
# #        os.environ[var] = getpass.getpass(f"{var}: ")
# #
# _set_env("OPENAI_API_KEY")
# #
# llm = ChatOpenAI(model="gpt-4o-mini")
#
# class State(TypedDict):
#     # Messages have the type "list". The `add_messages` function
#     # in the annotation defines how this state key should be updated
#     # (in this case, it appends messages to the list, rather than overwriting them)
#     messages: Annotated[list, add_messages]
#
#
# def chatbot(state: State):
#     return {"messages": [llm.invoke(state["messages"])]}
#
#
# class State(TypedDict):
#     # Messages have the type "list". The `add_messages` function
#     # in the annotation defines how this state key should be updated
#     # (in this case, it appends messages to the list, rather than overwriting them)
#     messages: Annotated[list, add_messages]
#
#
# graph_builder = StateGraph(State)
# graph_builder.add_node("chatbot2", chatbot)
# graph_builder.add_edge(START, "chatbot2")
# graph_builder.add_edge("chatbot2", END)
#
# graph = graph_builder.compile()
#
# img_bytes = graph.get_graph().draw_mermaid_png()
# output_path = Path("output/graph.png")
# output_path.parent.mkdir(parents=True, exist_ok=True)
#
# with open(output_path, "wb") as f:
#     f.write(img_bytes)
#
# print(f"Imagem salva em: {output_path.resolve()}")
#
# def stream_graph_updates(user_input: str):
#     for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)
#
#
# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break
#
#         stream_graph_updates(user_input)
#     except:
#         # fallback if input() is not available
#         user_input = "What do you know about LangGraph?"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, TypedDict
from langgraph.graph import StateGraph, END
import logging
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Carrega variáveis de ambiente (necessário para OPENAI_API_KEY)
load_dotenv()

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Funções Auxiliares (Carregamento - Parte 1 - sem mudanças) ---

def load_parquet_safe(file_path: Path) -> Optional[pd.DataFrame]:
    # ... (código da função load_parquet_safe como antes) ...
    if not file_path.is_file():
        logging.warning(f"Arquivo Parquet não encontrado em {file_path}")
        return None
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Carregado {file_path} com {len(df)} linhas.")
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar {file_path}: {e}")
        return None

def prepare_uc_origins(
    entities_df: Optional[pd.DataFrame],
    reports_df: Optional[pd.DataFrame]
) -> List[Dict[str, Any]]:
    # ... (código da função prepare_uc_origins como antes) ...
    uc_origins = []
    logging.info("Iniciando preparação das origens de UC...")

    # Processar Entidades
    if entities_df is not None:
        logging.info(f"Processando {len(entities_df)} entidades...")
        required_cols = ['id', 'title', 'description']
        if all(col in entities_df.columns for col in required_cols):
            for row in entities_df.itertuples(index=False):
                uc_origins.append({
                    "origin_id": row.id,
                    "origin_type": "entity",
                    "title": row.title,
                    "context": row.description if pd.notna(row.description) else ""
                })
        else:
            logging.warning(f"Colunas {required_cols} não encontradas em entities_df. Pulando processamento de entidades.")

    # Processar Resumos de Comunidades
    if reports_df is not None:
        logging.info(f"Processando {len(reports_df)} resumos de comunidade...")
        required_cols = ['id', 'community', 'title', 'summary']
        if all(col in reports_df.columns for col in required_cols):
            for row in reports_df.itertuples(index=False):
                uc_origins.append({
                    "origin_id": row.id,
                    "origin_type": "community_report",
                    "title": row.title,
                    "context": row.summary if pd.notna(row.summary) else "",
                    "community_human_id": row.community
                })
        else:
            required_cols_alt = ['community', 'title', 'summary']
            if all(col in reports_df.columns for col in required_cols_alt):
                 logging.warning("Coluna 'id' não encontrada em community_reports. Usando 'community' como origin_id (pode não ser UUID).")
                 for row in reports_df.itertuples(index=False):
                    uc_origins.append({
                        "origin_id": str(row.community),
                        "origin_type": "community_report",
                        "title": row.title,
                        "context": row.summary if pd.notna(row.summary) else "",
                        "community_human_id": row.community
                    })
            else:
                 logging.warning(f"Colunas {required_cols} (ou {required_cols_alt}) não encontradas em reports_df. Pulando processamento de reports.")

    logging.info(f"Total de {len(uc_origins)} origens preparadas para geração de UCs.")
    return uc_origins


# --- Definição do Estado do LangGraph (ATUALIZADO) ---

class UcGenerationState(TypedDict):
    """Define a estrutura do estado compartilhado no grafo."""
    graphrag_output_dir: Path
    uc_origins: List[Dict[str, Any]]
    generated_ucs: List[Dict[str, Any]] # NOVO: Para armazenar as UCs geradas
    error_message: Optional[str]

# --- Definição dos Nós do LangGraph ---

# Nó 1: Carregar e Preparar (sem mudanças na lógica interna)
def load_and_prepare_node(state: UcGenerationState) -> UcGenerationState:
    # ... (código do nó load_and_prepare_node como antes) ...
    logging.info("Executando o nó: load_and_prepare_node")
    output_dir = state.get("graphrag_output_dir")
    error_msg = state.get("error_message") # Preserva erros anteriores
    origins = []

    if not output_dir or not output_dir.is_dir():
        error_msg = f"Diretório de saída do GraphRAG não encontrado ou inválido: {output_dir}"
        logging.error(error_msg)
        # Retorna estado atualizado apenas com erro, mantendo o que já existia
        return {**state, "uc_origins": [], "error_message": error_msg}

    entities_file = output_dir / "entities.parquet"
    reports_file = output_dir / "community_reports.parquet"

    logging.info("Carregando dados do GraphRAG...")
    entities_data = load_parquet_safe(entities_file)
    reports_data = load_parquet_safe(reports_file)

    # Só tenta preparar se não houver erro anterior
    if not error_msg:
        try:
            logging.info("Preparando origens para UCs...")
            origins = prepare_uc_origins(entities_data, reports_data)
            if not origins:
                 logging.warning("Nenhuma origem de UC foi preparada. Verifique os arquivos Parquet.")
        except Exception as e:
            error_msg = f"Erro inesperado durante a preparação das origens: {e}"
            logging.exception(error_msg)
            origins = []

    # Atualiza o estado
    # Garante que 'generated_ucs' seja inicializado se ainda não existir
    current_generated_ucs = state.get("generated_ucs", [])
    return {**state, "uc_origins": origins, "generated_ucs": current_generated_ucs, "error_message": error_msg}


# Nó 2: Gerar UCs (NOVO)
def generate_ucs_node(state: UcGenerationState) -> UcGenerationState:
    """
    Nó do LangGraph que gera as UCs usando LLM para cada origem preparada.
    """
    logging.info("Executando o nó: generate_ucs_node")
    origins = state.get("uc_origins", [])
    error_msg = state.get("error_message")
    prompt_file_path = Path("prompt_uc_generation.txt")
    all_generated_ucs = [] # Lista para acumular UCs de todas as origens

    # Verifica erros anteriores ou falta de origens
    if error_msg:
        logging.warning(f"Pulando geração de UCs devido a erro anterior: {error_msg}")
        return state # Retorna estado inalterado (exceto talvez pelo erro já registrado)
    if not origins:
        logging.warning("Nenhuma origem de UC encontrada para processar. Pulando geração.")
        return state

    # Carrega o template do prompt
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        logging.info(f"Prompt carregado de {prompt_file_path}")
    except FileNotFoundError:
        error_msg = f"Arquivo de prompt não encontrado: {prompt_file_path}"
        logging.error(error_msg)
        return {**state, "generated_ucs": [], "error_message": error_msg}
    except Exception as e:
        error_msg = f"Erro ao ler arquivo de prompt: {e}"
        logging.error(error_msg)
        return {**state, "generated_ucs": [], "error_message": error_msg}

    # Inicializa o cliente LLM
    try:
        # Certifique-se de que a variável de ambiente OPENAI_API_KEY está definida
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2) # Usando gpt-4o-mini e baixa temperatura para consistência
        logging.info("Cliente ChatOpenAI inicializado com gpt-4o-mini.")
    except Exception as e:
        error_msg = f"Erro ao inicializar ChatOpenAI: {e}. Verifique sua API key."
        logging.error(error_msg)
        return {**state, "generated_ucs": [], "error_message": error_msg}

    # Itera sobre as origens e gera UCs
    logging.info(f"Iniciando geração de UCs para {len(origins)} origens...")
    for i, origin in enumerate(origins):
        origin_id = origin.get("origin_id")
        title = origin.get("title", "N/A")
        context = origin.get("context", "")

        logging.info(f"Processando origem {i+1}/{len(origins)}: ID={origin_id}, Título='{title[:50]}...'")

        # Formata o prompt específico para esta origem
        try:
            formatted_prompt = prompt_template.replace("{{CONCEPT_TITLE}}", title)
            formatted_prompt = formatted_prompt.replace("{{CONTEXT}}", context if context else "Nenhum contexto adicional fornecido.")
        except Exception as e:
            logging.error(f"Erro ao formatar prompt para origem {origin_id}: {e}")
            continue # Pula para a próxima origem

        # Define as mensagens para o LLM
        messages = [
            SystemMessage(content="Você é um assistente expert em educação que SEMPRE responde em formato JSON válido, conforme instruído."),
            HumanMessage(content=formatted_prompt),
        ]

        # Chama o LLM e processa a resposta
        try:
            response = llm.invoke(messages)
            response_content = response.content

            # Tenta parsear o JSON da resposta
            try:
                # Limpa possíveis ```json ``` do início/fim que alguns modelos adicionam
                if response_content.strip().startswith("```json"):
                    response_content = response_content.strip()[7:-3].strip()
                elif response_content.strip().startswith("```"):
                     response_content = response_content.strip()[3:-3].strip()

                data = json.loads(response_content)
                units = data.get("generated_units", [])

                if isinstance(units, list) and len(units) == 6:
                    logging.info(f"  Sucesso: 6 UCs recebidas para origem {origin_id}.")
                    # Adiciona o origin_id a cada UC gerada e armazena
                    for unit in units:
                        if isinstance(unit, dict) and "bloom_level" in unit and "uc_text" in unit:
                            unit["origin_id"] = origin_id # Vincula a UC à sua origem
                            all_generated_ucs.append(unit)
                        else:
                             logging.warning(f"  Formato inválido para uma UC da origem {origin_id}: {unit}")
                else:
                    logging.warning(f"  Resposta JSON recebida para origem {origin_id}, mas não continha a lista esperada de 6 UCs. Resposta: {response_content[:200]}...")

            except json.JSONDecodeError as json_err:
                logging.error(f"  Erro ao decodificar JSON da resposta LLM para origem {origin_id}: {json_err}. Resposta recebida: {response_content[:200]}...")
            except Exception as parse_err:
                 logging.error(f"  Erro inesperado ao processar JSON da origem {origin_id}: {parse_err}. Resposta: {response_content[:200]}...")


        except Exception as llm_err:
            logging.error(f"  Erro na chamada LLM para origem {origin_id}: {llm_err}")
            # Decide se quer parar ou continuar para as próximas origens
            # Por enquanto, vamos apenas logar e continuar

        # --- CONTROLE DE FLUXO (OPCIONAL): Para testes, processe apenas algumas origens ---
        # if i >= 4: # Processa apenas as 5 primeiras origens
        #     logging.warning("Limite de teste atingido. Interrompendo geração de UCs.")
        #     break
        # --------------------------------------------------------------------------------

    logging.info(f"Geração concluída. Total de {len(all_generated_ucs)} UCs individuais geradas.")

    # Atualiza o estado com as UCs geradas (mesmo que a lista esteja vazia)
    # Preserva outros campos do estado, incluindo error_message que pode ter sido definido antes
    return {**state, "generated_ucs": all_generated_ucs}


# --- Construção e Compilação do Grafo (ATUALIZADO) ---

workflow = StateGraph(UcGenerationState)

# Adiciona os nós
workflow.add_node("load_prepare", load_and_prepare_node)
workflow.add_node("generate_ucs", generate_ucs_node) # Adiciona o novo nó

# Define o ponto de entrada
workflow.set_entry_point("load_prepare")

# Define as transições entre os nós
workflow.add_edge("load_prepare", "generate_ucs") # Vai do nó 1 para o nó 2
workflow.add_edge("generate_ucs", END)           # Termina após o nó 2

# Compila o grafo
app = workflow.compile()

# --- Execução do Grafo ---

if __name__ == "__main__":
    GRAPH_RAG_OUTPUT_PATH = Path("./graphrag_outputs") # <-- AJUSTE ESTE CAMINHO

    initial_state: UcGenerationState = {
        "graphrag_output_dir": GRAPH_RAG_OUTPUT_PATH,
        "uc_origins": [],
        "generated_ucs": [], # Inicializa a nova chave do estado
        "error_message": None
    }

    print("--- Iniciando execução do grafo LangGraph (com Geração de UCs) ---")
    # Para fluxos mais longos, 'stream' pode ser útil para ver o estado após cada nó
    # final_state = None
    # for s in app.stream(initial_state):
    #     print(f"Estado após nó '{list(s.keys())[0]}':")
    #     # print(s) # Imprime todo o estado intermediário (pode ser grande)
    #     final_state = list(s.values())[0]

    # Usando invoke para obter apenas o estado final
    final_state = app.invoke(initial_state)

    print("--- Execução do grafo LangGraph concluída ---")

    # --- Verificação do Resultado ---
    if final_state.get("error_message"):
        print(f"\nERRO DURANTE A EXECUÇÃO: {final_state['error_message']}")

    generated_ucs_result = final_state.get("generated_ucs", [])
    if generated_ucs_result:
        print(f"\nSucesso! {len(generated_ucs_result)} UCs individuais foram geradas.")
        print("\nExemplo das primeiras 12 UCs geradas (2 origens):")
        for i, uc in enumerate(generated_ucs_result[:12]):
             print(f"--- UC {i+1} ---")
             print(f"  Origem ID: {uc.get('origin_id')}")
             print(f"  Nível Bloom: {uc.get('bloom_level')}")
             print(f"  Texto UC: {uc.get('uc_text')}")
    else:
        if not final_state.get("error_message"): # Só mostra se não houve erro fatal antes
            print("\nNenhuma UC foi gerada. Verifique os logs para detalhes.")
            origins_count = len(final_state.get("uc_origins", []))
            if origins_count == 0:
                print("  (Também não foram encontradas origens na etapa anterior)")