from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Set
from sqlalchemy import or_

import app.crud as crud
import app.models as models
import app.schemas as schemas
from app.db import get_db

router = APIRouter(
    prefix="/api/v1",
    tags=["Results"],
)

# --- Endpoints para Pipeline Runs (Básico) ---
@router.get("/runs", response_model=List[schemas.PipelineRunSummary])
def list_pipeline_runs(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db)
):
    runs = db.query(models.PipelineRun).order_by(models.PipelineRun.started_at.desc()).offset(skip).limit(limit).all()
    return runs


@router.get("/runs/{run_id}", response_model=schemas.PipelineRunDetail)
def get_pipeline_run_details(
        run_id: str,
        db: Session = Depends(get_db)
):
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
    return run

# --- Endpoints para Resultados do Grafo de Conhecimento Educacional ---

@router.get(
    "/runs/{run_id}/knowledge-units",
    response_model=List[schemas.FinalKnowledgeUnitResponse],
    summary="List Final Knowledge Units for a Run"
)
def list_final_knowledge_units(
        run_id: str,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        bloom_level: Optional[str] = Query(None, description="Filter by Bloom's taxonomy level"),
        origin_id: Optional[str] = Query(None, description="Filter by origin ID"),
        min_difficulty: Optional[int] = Query(None, ge=0, le=100),
        max_difficulty: Optional[int] = Query(None, ge=0, le=100),
        db: Session = Depends(get_db)
):
    """
    Retrieves a list of final knowledge units (UCs) for a specific pipeline run,
    with optional filtering and pagination.
    """
    # Verifica se a run existe
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    query = db.query(models.FinalKnowledgeUnit).filter(models.FinalKnowledgeUnit.pipeline_run_id == run_id)

    if bloom_level:
        query = query.filter(models.FinalKnowledgeUnit.bloom_level == bloom_level)
    if origin_id:
        query = query.filter(models.FinalKnowledgeUnit.origin_id == origin_id)
    if min_difficulty is not None:
        query = query.filter(models.FinalKnowledgeUnit.difficulty_score >= min_difficulty)
    if max_difficulty is not None:
        query = query.filter(models.FinalKnowledgeUnit.difficulty_score <= max_difficulty)

    units = query.offset(skip).limit(limit).all()
    return units


@router.get(
    "/runs/{run_id}/knowledge-units/{uc_id}",
    response_model=schemas.FinalKnowledgeUnitResponse,
    summary="Get a Specific Final Knowledge Unit"
)
def get_final_knowledge_unit(
        run_id: str,
        uc_id: str,
        db: Session = Depends(get_db)
):
    """
    Retrieves details for a specific final knowledge unit (UC) within a pipeline run.
    """
    unit = db.query(models.FinalKnowledgeUnit).filter(
        models.FinalKnowledgeUnit.pipeline_run_id == run_id,
        models.FinalKnowledgeUnit.uc_id == uc_id
    ).first()

    if not unit:
        raise HTTPException(status_code=404, detail=f"Knowledge Unit '{uc_id}' not found in run '{run_id}'")
    return unit


@router.get(
    "/runs/{run_id}/relationships",
    response_model=List[schemas.FinalKnowledgeRelationshipResponse],
    summary="List Final Knowledge Relationships for a Run"
)
def list_final_relationships(
        run_id: str,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        type: Optional[str] = Query(None, description="Filter by relationship type (e.g., REQUIRES, EXPANDS)"),

        uc_id_involved: Optional[str] = Query(None, description="Filter by UC ID involved as source or target"),
        source_uc_id: Optional[str] = Query(None, description="Filter by source UC ID"),
        target_uc_id: Optional[str] = Query(None, description="Filter by target UC ID"),
        origin_id: Optional[str] = Query(None, description="Filter by relationship's origin ID"),
        db: Session = Depends(get_db)
):
    """
    Retrieves a list of final knowledge relationships for a specific pipeline run,
    with optional filtering and pagination.
    """
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    query = db.query(models.FinalKnowledgeRelationship).filter(
        models.FinalKnowledgeRelationship.pipeline_run_id == run_id)

    if type:
        query = query.filter(models.FinalKnowledgeRelationship.type == type)
    
    if uc_id_involved:
        query = query.filter(
            or_(
                models.FinalKnowledgeRelationship.source == uc_id_involved,
                models.FinalKnowledgeRelationship.target == uc_id_involved
            )
        )
    
    if source_uc_id:
        query = query.filter(models.FinalKnowledgeRelationship.source == source_uc_id)

    if target_uc_id:
        query = query.filter(models.FinalKnowledgeRelationship.target == target_uc_id)

    if origin_id:
        query = query.filter(models.FinalKnowledgeRelationship.origin_id == origin_id)


    # Adicionar ordenação se desejar
    # query = query.order_by(models.FinalKnowledgeRelationship.source)

    relationships = query.offset(skip).limit(limit).all()
    return relationships


# --- Endpoints para Origens das UCs ---
@router.get(
    "/runs/{run_id}/origins",
    response_model=List[schemas.KnowledgeUnitOriginResponse],
    summary="List Knowledge Unit Origins for a Run"
)
def list_knowledge_unit_origins(
        run_id: str,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        origin_type: Optional[str] = Query(None, description="Filter by origin type (e.g., entity, community_summary)"),
        db: Session = Depends(get_db)
):
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    query = db.query(models.KnowledgeUnitOrigin).filter(models.KnowledgeUnitOrigin.pipeline_run_id == run_id)
    if origin_type:
        query = query.filter(models.KnowledgeUnitOrigin.origin_type == origin_type)

    origins = query.offset(skip).limit(limit).all()
    return origins

@router.get(
    "/runs/{run_id}/origins/{origin_id}/knowledge-units",
    response_model=List[schemas.FinalKnowledgeUnitResponse],
    summary="List Final Knowledge Units for a Specific Origin"
)
def list_knowledge_units_for_origin(
        run_id: str,
        origin_id: str,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        db: Session = Depends(get_db)
):
    # Verifica se a run e a origin existem
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    origin_exists = db.query(models.KnowledgeUnitOrigin).filter_by(pipeline_run_id=run_id, origin_id=origin_id).first()
    if not origin_exists:
        raise HTTPException(status_code=404, detail=f"Origin '{origin_id}' not found in run '{run_id}'")

    query = db.query(models.FinalKnowledgeUnit).filter(
        models.FinalKnowledgeUnit.pipeline_run_id == run_id,
        models.FinalKnowledgeUnit.origin_id == origin_id
    )

    units = query.offset(skip).limit(limit).all()
    return units

# --- Endpoints para Visualização/Exploração do Grafo ---

def format_uc_as_graph_node(uc: models.FinalKnowledgeUnit) -> schemas.GraphNode:
    """Helper para converter uma UC do DB para um nó de grafo."""
    return schemas.GraphNode(
        id=uc.uc_id,
        label=uc.uc_text[:50] + "..." if uc.uc_text and len(uc.uc_text) > 50 else uc.uc_text,
        title=uc.uc_text,
        group=uc.origin_id or "unknown_origin",
        level=uc.bloom_level,
        value=uc.difficulty_score
    )

def format_rel_as_graph_edge(rel: models.FinalKnowledgeRelationship) -> schemas.GraphEdge:
    """Helper para converter uma Relação do DB para uma aresta de grafo."""
    return schemas.GraphEdge(
        source=rel.source,
        target=rel.target,
        label=rel.type
    )

@router.get(
    "/runs/{run_id}/graph/sample",
    response_model=schemas.GraphResponse,
    summary="Get a Sample of the Knowledge Graph"
)
def get_graph_sample(
        run_id: str,
        sample_size: int = Query(50, ge=10, le=200, description="Approximate number of nodes in the sample"),
        db: Session = Depends(get_db)
):
    """
    Retrieves a sample of the knowledge graph for a specific pipeline run.
    This is a simplified sample, e.g., first N UCs and their direct relationships.
    """
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    sampled_ucs_db = db.query(models.FinalKnowledgeUnit)\
                       .filter(models.FinalKnowledgeUnit.pipeline_run_id == run_id)\
                       .limit(sample_size)\
                       .all()

    if not sampled_ucs_db:
        return schemas.GraphResponse(nodes=[], edges=[])

    sampled_uc_ids = {uc.uc_id for uc in sampled_ucs_db}
    
    nodes: List[schemas.GraphNode] = [format_uc_as_graph_node(uc) for uc in sampled_ucs_db]
    
    sampled_rels_db = db.query(models.FinalKnowledgeRelationship)\
                        .filter(
                            models.FinalKnowledgeRelationship.pipeline_run_id == run_id,
                            models.FinalKnowledgeRelationship.source.in_(sampled_uc_ids),
                            models.FinalKnowledgeRelationship.target.in_(sampled_uc_ids)
                        ).all()
    
    edges: List[schemas.GraphEdge] = [format_rel_as_graph_edge(rel) for rel in sampled_rels_db]

    return schemas.GraphResponse(nodes=nodes, edges=edges)


@router.get(
    "/runs/{run_id}/graph/neighborhood/{uc_id}",
    response_model=schemas.GraphResponse,
    summary="Get Neighborhood of a Specific Knowledge Unit"
)
def get_uc_neighborhood(
        run_id: str,
        uc_id: str,
        depth: int = Query(1, ge=1, le=3, description="Number of hops from the central UC (1 = direct neighbors)"),
        db: Session = Depends(get_db)
):
    """
    Retrieves a specific Knowledge Unit and its neighbors within a certain depth.
    """
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    central_uc_db = db.query(models.FinalKnowledgeUnit).filter(
        models.FinalKnowledgeUnit.pipeline_run_id == run_id,
        models.FinalKnowledgeUnit.uc_id == uc_id
    ).first()

    if not central_uc_db:
        raise HTTPException(status_code=404, detail=f"Central UC '{uc_id}' not found in run '{run_id}'")

    nodes_map = {central_uc_db.uc_id: format_uc_as_graph_node(central_uc_db)}
    edges_list: List[schemas.GraphEdge] = []
    
    current_level_uc_ids: Set[str] = {uc_id}
    all_included_uc_ids: Set[str] = {uc_id}

    for _ in range(depth):
        if not current_level_uc_ids:
            break

        next_level_uc_ids: Set[str] = set()
        
        rels_db = db.query(models.FinalKnowledgeRelationship).filter(
            models.FinalKnowledgeRelationship.pipeline_run_id == run_id,
            or_(
                models.FinalKnowledgeRelationship.source.in_(current_level_uc_ids),
                models.FinalKnowledgeRelationship.target.in_(current_level_uc_ids)
            )
        ).all()

        newly_added_neighbor_ids: Set[str] = set()

        for rel in rels_db:
            edges_list.append(format_rel_as_graph_edge(rel))
            
            neighbor_id = None
            if rel.source in current_level_uc_ids and rel.target not in all_included_uc_ids:
                neighbor_id = rel.target
            elif rel.target in current_level_uc_ids and rel.source not in all_included_uc_ids:
                neighbor_id = rel.source
            
            if neighbor_id:
                newly_added_neighbor_ids.add(neighbor_id)
                all_included_uc_ids.add(neighbor_id)
        
        if not newly_added_neighbor_ids:
            break

        neighbor_ucs_db = db.query(models.FinalKnowledgeUnit).filter(
            models.FinalKnowledgeUnit.pipeline_run_id == run_id,
            models.FinalKnowledgeUnit.uc_id.in_(newly_added_neighbor_ids)
        ).all()

        for uc in neighbor_ucs_db:
            nodes_map[uc.uc_id] = format_uc_as_graph_node(uc)
            next_level_uc_ids.add(uc.uc_id)
            
        current_level_uc_ids = next_level_uc_ids

    return schemas.GraphResponse(nodes=list(nodes_map.values()), edges=edges_list)



@router.get(
    "/runs/{run_id}/origins-hierarchy/roots",
    response_model=List[schemas.KnowledgeUnitOriginResponse],  # Usando o schema que já existe para listar origens
    summary="List Root-Level Knowledge Unit Origins"
)
def list_root_origins(
        run_id: str,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        origin_type: Optional[str] = Query(None, description="Filter root origins by type (e.g., entity, community)"),
        db: Session = Depends(get_db)
):
    """
    Retrieves a list of root-level knowledge unit origins (those without a parent)
    for a specific pipeline run.
    """
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    query = db.query(models.KnowledgeUnitOrigin).filter(
        models.KnowledgeUnitOrigin.pipeline_run_id == run_id,
        # Verifica se parent_community_id_of_origin é NULL ou uma string vazia,
        # dependendo de como você armazena "sem pai".
        # SQLAlchemy trata None como IS NULL. Se você usa string vazia, ajuste.
        models.KnowledgeUnitOrigin.parent_community_id_of_origin == None
    )

    if origin_type:
        query = query.filter(models.KnowledgeUnitOrigin.origin_type == origin_type)

    # Adicionar ordenação se desejar, ex: por title
    query = query.order_by(models.KnowledgeUnitOrigin.title)

    roots = query.offset(skip).limit(limit).all()
    return roots


@router.get(
    "/runs/{run_id}/origins-hierarchy/children/{parent_origin_id}",
    response_model=List[schemas.KnowledgeUnitOriginResponse],
    summary="List Child Knowledge Unit Origins"
)
def list_child_origins(
        run_id: str,
        parent_origin_id: str,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        origin_type: Optional[str] = Query(None, description="Filter child origins by type (e.g., entity, community)"),
        db: Session = Depends(get_db)
):
    """
    Retrieves a list of direct child knowledge unit origins for a given parent origin_id
    within a specific pipeline run.
    """
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    # Opcional: Verificar se o parent_origin_id existe para esta run
    # parent_exists = db.query(models.KnowledgeUnitOrigin).filter_by(pipeline_run_id=run_id, origin_id=parent_origin_id).first()
    # if not parent_exists:
    #     raise HTTPException(status_code=404, detail=f"Parent origin '{parent_origin_id}' not found in run '{run_id}'")

    query = db.query(models.KnowledgeUnitOrigin).filter(
        models.KnowledgeUnitOrigin.pipeline_run_id == run_id,
        models.KnowledgeUnitOrigin.parent_community_id_of_origin == parent_origin_id
    )

    if origin_type:
        query = query.filter(models.KnowledgeUnitOrigin.origin_type == origin_type)

    # Adicionar ordenação se desejar, ex: por title
    query = query.order_by(models.KnowledgeUnitOrigin.title)

    children = query.offset(skip).limit(limit).all()
    return children


@router.get(
    "/runs/{run_id}/origins-hierarchy/parent/{child_origin_id}",
    response_model=Optional[schemas.KnowledgeUnitOriginResponse],  # Pode não ter pai, então Optional
    summary="Get Parent of a Knowledge Unit Origin"
)
def get_parent_origin(
        run_id: str,
        child_origin_id: str,
        db: Session = Depends(get_db)
):
    """
    Retrieves the parent knowledge unit origin for a given child_origin_id
    within a specific pipeline run. Returns null if the origin has no parent or is not found.
    """
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    child_origin = db.query(models.KnowledgeUnitOrigin).filter(
        models.KnowledgeUnitOrigin.pipeline_run_id == run_id,
        models.KnowledgeUnitOrigin.origin_id == child_origin_id
    ).first()

    if not child_origin:
        raise HTTPException(status_code=404, detail=f"Child origin '{child_origin_id}' not found in run '{run_id}'")

    if not child_origin.parent_community_id_of_origin:
        return None  # Origem é uma raiz, não tem pai

    parent_origin = db.query(models.KnowledgeUnitOrigin).filter(
        models.KnowledgeUnitOrigin.pipeline_run_id == run_id,
        models.KnowledgeUnitOrigin.origin_id == child_origin.parent_community_id_of_origin
    ).first()

    # Se parent_community_id_of_origin tiver um valor mas não encontrarmos uma origem com esse ID,
    # isso indicaria uma inconsistência nos dados. Por ora, retornamos None ou um 404.
    if not parent_origin:
        # Isso pode acontecer se o parent_community_id_of_origin for um ID que não existe como uma 'origin' na tabela.
        # O ideal é que todos os parent_community_id_of_origin (que não são None) referenciem um origin_id válido.
        # raise HTTPException(status_code=404, detail=f"Parent origin with ID '{child_origin.parent_community_id_of_origin}' not found, data inconsistency suspected.")
        return None  # Ou tratar como "não encontrado" para simplificar

    return parent_origin





def format_origin_as_graph_node(origin: models.KnowledgeUnitOrigin) -> schemas.GraphNode: # Reutilizando GraphNode por simplicidade
    """Converte um KnowledgeUnitOrigin para um formato de nó de grafo."""
    return schemas.GraphNode(
        id=origin.origin_id,
        label=origin.title[:75] + "..." if origin.title and len(origin.title) > 75 else origin.title,
        group=origin.origin_type, # "entity" ou "community"
        title=origin.title, # Tooltip com título completo
        level=str(origin.level) if origin.level is not None else None, # Nível da comunidade, se aplicável
        # value=origin.frequency or origin.degree # Exemplo, se quiser dimensionar
    )


@router.get(
    "/runs/{run_id}/origins-hierarchy/tree/{start_origin_id}",
    response_model=schemas.GraphResponse,  # Reutilizando GraphResponse
    summary="Get a Sub-tree of the Origins Hierarchy"
)
def get_origins_hierarchy_tree(
        run_id: str,
        start_origin_id: str,
        depth: int = Query(2, ge=0, le=5,
                           description="Number of levels to descend from the start_origin_id (0 = just the start node)"),
        db: Session = Depends(get_db)
):
    """
    Retrieves a sub-tree of the knowledge unit origins hierarchy,
    starting from a specific origin_id and descending N levels.
    """
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    start_origin_db = db.query(models.KnowledgeUnitOrigin).filter(
        models.KnowledgeUnitOrigin.pipeline_run_id == run_id,
        models.KnowledgeUnitOrigin.origin_id == start_origin_id
    ).first()

    if not start_origin_db:
        raise HTTPException(status_code=404, detail=f"Start origin '{start_origin_id}' not found in run '{run_id}'")

    nodes_map = {start_origin_db.origin_id: format_origin_as_graph_node(start_origin_db)}
    edges_list: List[schemas.GraphEdge] = []  # Reutilizando GraphEdge

    # IDs das origens a serem processadas no nível atual da árvore
    current_level_origin_ids: Set[str] = {start_origin_id}

    # Itera pela profundidade desejada
    for current_depth in range(depth):
        if not current_level_origin_ids:  # Se não há mais nós para expandir
            break

        # Busca todos os filhos diretos das origens no nível atual
        children_db = db.query(models.KnowledgeUnitOrigin).filter(
            models.KnowledgeUnitOrigin.pipeline_run_id == run_id,
            models.KnowledgeUnitOrigin.parent_community_id_of_origin.in_(current_level_origin_ids)
        ).all()

        next_level_origin_ids: Set[str] = set()

        for child in children_db:
            # Adiciona o nó filho se ainda não estiver no mapa
            if child.origin_id not in nodes_map:
                nodes_map[child.origin_id] = format_origin_as_graph_node(child)

            # Adiciona a aresta do pai (que deve estar em current_level_origin_ids) para o filho
            # O pai é child.parent_community_id_of_origin
            if child.parent_community_id_of_origin:  # Garantir que tem pai
                edges_list.append(schemas.GraphEdge(
                    source=child.parent_community_id_of_origin,
                    target=child.origin_id,
                    label="contains"  # Ou "child_of", ou deixar sem label
                ))

            next_level_origin_ids.add(child.origin_id)  # Adiciona para a próxima iteração

        current_level_origin_ids = next_level_origin_ids  # Prepara para o próximo nível

    return schemas.GraphResponse(nodes=list(nodes_map.values()), edges=edges_list)