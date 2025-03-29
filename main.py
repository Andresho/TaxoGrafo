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

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Funções Auxiliares (do script anterior) ---

def load_parquet_safe(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Carrega um arquivo Parquet de forma segura, retornando None se não existir ou erro.
    """
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
    """
    Prepara a lista de 'origens' para a geração de UCs a partir dos DataFrames.
    """
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

# --- Definição do Estado do LangGraph ---

class UcGenerationState(TypedDict):
    """Define a estrutura do estado compartilhado no grafo."""
    graphrag_output_dir: Path       # Diretório de entrada com os Parquets
    uc_origins: List[Dict[str, Any]] # Resultado: Lista de origens prontas
    error_message: Optional[str]    # Para registrar erros durante a execução

# --- Definição do Nó do LangGraph ---

def load_and_prepare_node(state: UcGenerationState) -> UcGenerationState:
    """
    Nó do LangGraph que carrega os dados do GraphRAG e prepara as origens de UC.
    """
    logging.info("Executando o nó: load_and_prepare_node")
    output_dir = state.get("graphrag_output_dir")
    error_msg = None
    origins = []

    if not output_dir or not output_dir.is_dir():
        error_msg = f"Diretório de saída do GraphRAG não encontrado ou inválido: {output_dir}"
        logging.error(error_msg)
        return {**state, "uc_origins": [], "error_message": error_msg}

    entities_file = output_dir / "entities.parquet"
    reports_file = output_dir / "community_reports.parquet"

    logging.info("Carregando dados do GraphRAG...")
    entities_data = load_parquet_safe(entities_file)
    reports_data = load_parquet_safe(reports_file)

    try:
        logging.info("Preparando origens para UCs...")
        origins = prepare_uc_origins(entities_data, reports_data)
        if not origins:
             logging.warning("Nenhuma origem de UC foi preparada. Verifique os arquivos Parquet.")
             # Não definimos como erro fatal, mas pode ser necessário dependendo do fluxo
    except Exception as e:
        error_msg = f"Erro inesperado durante a preparação das origens: {e}"
        logging.exception(error_msg) # Loga o traceback completo
        origins = [] # Garante que origins esteja vazia em caso de erro

    # Atualiza o estado com os resultados ou erros
    return {**state, "uc_origins": origins, "error_message": error_msg}

# --- Construção e Compilação do Grafo ---

# Instancia o grafo com o estado definido
workflow = StateGraph(UcGenerationState)

# Adiciona o nó ao grafo
workflow.add_node("load_prepare", load_and_prepare_node)

# Define o ponto de entrada do grafo
workflow.set_entry_point("load_prepare")

# Define o ponto final (por enquanto, termina após o primeiro nó)
# Futuramente, adicionaremos arestas para os próximos nós (geração de UC, etc.)
workflow.add_edge("load_prepare", END)

# Compila o grafo em uma aplicação executável
app = workflow.compile()

img_bytes = app.get_graph().draw_mermaid_png()
output_path = Path("output/graph.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "wb") as f:
    f.write(img_bytes)

print(f"Imagem salva em: {output_path.resolve()}")

# --- Execução do Grafo ---

if __name__ == "__main__":
    # Defina o diretório onde os arquivos Parquet do GraphRAG foram salvos
    GRAPH_RAG_OUTPUT_PATH = Path("./graphrag_outputs") # <-- AJUSTE ESTE CAMINHO se necessário

    # Define o estado inicial para a execução
    initial_state: UcGenerationState = {
        "graphrag_output_dir": GRAPH_RAG_OUTPUT_PATH,
        "uc_origins": [], # Inicializa vazio
        "error_message": None
    }

    print("--- Iniciando execução do grafo LangGraph ---")
    # Invoca o grafo com o estado inicial
    # O método stream retorna um gerador com os estados intermediários,
    # mas para um grafo simples, invoke é suficiente para obter o estado final.
    final_state = app.invoke(initial_state)
    print("--- Execução do grafo LangGraph concluída ---")

    # --- Verificação do Resultado ---
    if final_state.get("error_message"):
        print(f"\nERRO DURANTE A EXECUÇÃO: {final_state['error_message']}")
    else:
        prepared_origins = final_state.get("uc_origins", [])
        if prepared_origins:
            print(f"\nSucesso! {len(prepared_origins)} origens de UC foram preparadas.")
            print("\nExemplo das primeiras 5 origens:")
            for i, origem in enumerate(prepared_origins[:5]):
                print(f"--- Origem {i+1} ---")
                print(f"  ID: {origem['origin_id']}")
                print(f"  Tipo: {origem['origin_type']}")
                print(f"  Título: {origem['title']}")
                # print(f"  Contexto: {origem['context'][:100]}...") # Descomente para ver
                if 'community_human_id' in origem:
                    print(f"  Community ID: {origem['community_human_id']}")
        else:
            print("\nA execução terminou sem erros, mas nenhuma origem de UC foi gerada.")
            print("Verifique os logs e os arquivos Parquet no diretório de entrada.")

