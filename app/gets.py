from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Set, Dict
from sqlalchemy import or_

import app.crud as crud
import app.models as models
import app.schemas as schemas
from app.db import get_db

import logging

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


@router.get(
    "/runs/{run_id}/origins/detailed",
    response_model=List[schemas.OriginWithUCsAndRelationshipsResponse],
    summary="List Knowledge Unit Origins with their UCs and Relationships (including related UC difficulty)"
)
def list_detailed_knowledge_unit_origins(
        run_id: str,
        skip: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=1000),
        origin_type_filter: Optional[str] = Query(None, alias="origin_type"),
        db: Session = Depends(get_db)
):
    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    query_origins = db.query(models.KnowledgeUnitOrigin).filter(
        models.KnowledgeUnitOrigin.pipeline_run_id == run_id
    )
    if origin_type_filter:
        query_origins = query_origins.filter(models.KnowledgeUnitOrigin.origin_type == origin_type_filter)

    db_origins = query_origins.order_by(models.KnowledgeUnitOrigin.title).offset(skip).limit(limit).all()

    response_list: List[schemas.OriginWithUCsAndRelationshipsResponse] = []
    uc_model_cache: Dict[str, models.FinalKnowledgeUnit] = {}  # Cache para modelos de UCs

    for db_origin in db_origins:
        origin_response_data = schemas.OriginWithUCsAndRelationshipsResponse.model_validate(db_origin).model_dump()
        origin_response_data["knowledge_units"] = []

        db_ucs_for_origin = db.query(models.FinalKnowledgeUnit).filter(
            models.FinalKnowledgeUnit.pipeline_run_id == run_id,
            models.FinalKnowledgeUnit.origin_id == db_origin.origin_id
        ).all()

        for db_uc in db_ucs_for_origin:
            if db_uc.uc_id not in uc_model_cache:
                uc_model_cache[db_uc.uc_id] = db_uc

            uc_response_base_data = schemas.FinalKnowledgeUnitResponse.model_validate(db_uc).model_dump()

            current_uc_relationships_as_source: List[schemas.UCRelationshipDetail] = []
            current_uc_relationships_as_target: List[schemas.UCRelationshipDetail] = []

            # Popular relationships_as_source
            db_rels_as_source = db.query(models.FinalKnowledgeRelationship).filter(
                models.FinalKnowledgeRelationship.pipeline_run_id == run_id,
                models.FinalKnowledgeRelationship.source == db_uc.uc_id
            ).all()

            for db_rel_s in db_rels_as_source:
                related_text = None
                related_bloom_level = None
                related_difficulty_score = None

                target_uc_model = uc_model_cache.get(db_rel_s.target)
                if not target_uc_model:
                    target_uc_model = db.query(models.FinalKnowledgeUnit).filter_by(
                        pipeline_run_id=run_id, uc_id=db_rel_s.target
                    ).first()
                    if target_uc_model:
                        uc_model_cache[db_rel_s.target] = target_uc_model

                if target_uc_model:
                    related_text = target_uc_model.uc_text
                    related_bloom_level = target_uc_model.bloom_level
                    related_difficulty_score = target_uc_model.difficulty_score

                current_uc_relationships_as_source.append(
                    schemas.UCRelationshipDetail(
                        relationship_type=db_rel_s.type,
                        related_uc_id=db_rel_s.target,
                        related_uc_text=related_text,
                        related_uc_bloom_level=related_bloom_level,
                        related_uc_difficulty_score=related_difficulty_score
                    )
                )

            # Popular relationships_as_target
            db_rels_as_target = db.query(models.FinalKnowledgeRelationship).filter(
                models.FinalKnowledgeRelationship.pipeline_run_id == run_id,
                models.FinalKnowledgeRelationship.target == db_uc.uc_id
            ).all()

            for db_rel_t in db_rels_as_target:
                related_text = None
                related_bloom_level = None
                related_difficulty_score = None

                source_uc_model = uc_model_cache.get(db_rel_t.source)
                if not source_uc_model:
                    source_uc_model = db.query(models.FinalKnowledgeUnit).filter_by(
                        pipeline_run_id=run_id, uc_id=db_rel_t.source
                    ).first()
                    if source_uc_model:
                        uc_model_cache[db_rel_t.source] = source_uc_model

                if source_uc_model:
                    related_text = source_uc_model.uc_text
                    related_bloom_level = source_uc_model.bloom_level
                    related_difficulty_score = source_uc_model.difficulty_score

                current_uc_relationships_as_target.append(
                    schemas.UCRelationshipDetail(
                        relationship_type=db_rel_t.type,
                        related_uc_id=db_rel_t.source,
                        related_uc_text=related_text,
                        related_uc_bloom_level=related_bloom_level,
                        related_uc_difficulty_score=related_difficulty_score
                    )
                )

            uc_with_rels_obj = schemas.UCWithRelationshipsResponse(
                **uc_response_base_data,
                relationships_as_source=current_uc_relationships_as_source,
                relationships_as_target=current_uc_relationships_as_target
            )
            origin_response_data["knowledge_units"].append(uc_with_rels_obj)

        final_origin_response_obj = schemas.OriginWithUCsAndRelationshipsResponse(**origin_response_data)
        response_list.append(final_origin_response_obj)

    return response_list


uc_model_cache_for_request: Dict[str, models.FinalKnowledgeUnit] = {}


def _build_detailed_origin_node(
        db_origin: models.KnowledgeUnitOrigin,
        run_id: str,
        db: Session,
        max_depth: int,
        limit_children_per_node: Optional[int],  # NOVO PARÂMETRO
        current_depth: int = 0
) -> schemas.OriginWithUCsAndRelationshipsResponse:
    """
    Constrói um nó da árvore de Origin detalhado, incluindo UCs, relações e filhos recursivamente.
    """
    global uc_model_cache_for_request  # Acessa o cache global da requisição

    logging.debug(f"Construindo nó para Origin ID: {db_origin.origin_id} na profundidade: {current_depth}")

    # Usar model_validate para criar o dicionário base a partir do objeto ORM
    origin_response_data = schemas.OriginWithUCsAndRelationshipsResponse.model_validate(db_origin).model_dump()
    origin_response_data["knowledge_units"] = []  # Inicializa a lista de UCs
    origin_response_data["children"] = []  # Inicializa a lista de filhos

    # 1. Popular Knowledge Units e suas Relações
    db_ucs_for_origin = db.query(models.FinalKnowledgeUnit).filter(
        models.FinalKnowledgeUnit.pipeline_run_id == run_id,
        models.FinalKnowledgeUnit.origin_id == db_origin.origin_id
    ).all()

    for db_uc_item in db_ucs_for_origin:  # Renomeado db_uc para db_uc_item
        if db_uc_item.uc_id not in uc_model_cache_for_request:
            uc_model_cache_for_request[db_uc_item.uc_id] = db_uc_item

        # Montar os dados base da UC
        uc_response_base_data = schemas.FinalKnowledgeUnitResponse.model_validate(db_uc_item).model_dump()

        current_uc_relationships_as_source: List[schemas.UCRelationshipDetail] = []
        current_uc_relationships_as_target: List[schemas.UCRelationshipDetail] = []

        # Popular relationships_as_source
        db_rels_as_source = db.query(models.FinalKnowledgeRelationship).filter(
            models.FinalKnowledgeRelationship.pipeline_run_id == run_id,
            models.FinalKnowledgeRelationship.source == db_uc_item.uc_id
        ).all()
        for db_rel_s in db_rels_as_source:
            related_text, related_bloom, related_difficulty = None, None, None
            target_uc_model = uc_model_cache_for_request.get(db_rel_s.target)
            if not target_uc_model:
                target_uc_model = db.query(models.FinalKnowledgeUnit).filter_by(pipeline_run_id=run_id,
                                                                                uc_id=db_rel_s.target).first()
                if target_uc_model: uc_model_cache_for_request[db_rel_s.target] = target_uc_model
            if target_uc_model:
                related_text, related_bloom, related_difficulty = target_uc_model.uc_text, target_uc_model.bloom_level, target_uc_model.difficulty_score
            current_uc_relationships_as_source.append(schemas.UCRelationshipDetail(
                relationship_type=db_rel_s.type, related_uc_id=db_rel_s.target,
                related_uc_text=related_text, related_uc_bloom_level=related_bloom,
                related_uc_difficulty_score=related_difficulty
            ))

        # Popular relationships_as_target
        db_rels_as_target = db.query(models.FinalKnowledgeRelationship).filter(
            models.FinalKnowledgeRelationship.pipeline_run_id == run_id,
            models.FinalKnowledgeRelationship.target == db_uc_item.uc_id
        ).all()
        for db_rel_t in db_rels_as_target:
            related_text, related_bloom, related_difficulty = None, None, None
            source_uc_model = uc_model_cache_for_request.get(db_rel_t.source)
            if not source_uc_model:
                source_uc_model = db.query(models.FinalKnowledgeUnit).filter_by(pipeline_run_id=run_id,
                                                                                uc_id=db_rel_t.source).first()
                if source_uc_model: uc_model_cache_for_request[db_rel_t.source] = source_uc_model
            if source_uc_model:
                related_text, related_bloom, related_difficulty = source_uc_model.uc_text, source_uc_model.bloom_level, source_uc_model.difficulty_score
            current_uc_relationships_as_target.append(schemas.UCRelationshipDetail(
                relationship_type=db_rel_t.type, related_uc_id=db_rel_t.source,
                related_uc_text=related_text, related_uc_bloom_level=related_bloom,
                related_uc_difficulty_score=related_difficulty
            ))

        # Criar o objeto UCWithRelationshipsResponse completo
        uc_with_rels_obj = schemas.UCWithRelationshipsResponse(
            **uc_response_base_data,  # Campos da FinalKnowledgeUnitResponse
            relationships_as_source=current_uc_relationships_as_source,
            relationships_as_target=current_uc_relationships_as_target
        )
        origin_response_data["knowledge_units"].append(uc_with_rels_obj)

    # 2. Popular Filhos (Recursivamente com limite de largura)
    if current_depth < max_depth:
        logging.debug(
            f"Buscando filhos para Origin ID: {db_origin.origin_id} (profundidade atual {current_depth}, max {max_depth}, limite filhos: {limit_children_per_node})")

        query_children = db.query(models.KnowledgeUnitOrigin).filter(
            models.KnowledgeUnitOrigin.pipeline_run_id == run_id,
            models.KnowledgeUnitOrigin.parent_community_id_of_origin == db_origin.origin_id
        ).order_by(models.KnowledgeUnitOrigin.title)  # Ordenar para consistência

        if limit_children_per_node is not None and limit_children_per_node > 0:  # Aplicar limite de filhos
            logging.debug(f"Aplicando limite de {limit_children_per_node} filhos para Origin ID: {db_origin.origin_id}")
            query_children = query_children.limit(limit_children_per_node)

        child_origins_db = query_children.all()

        if child_origins_db:
            logging.info(
                f"Origin {db_origin.origin_id} tem {len(child_origins_db)} filhos (após limite de {limit_children_per_node}). Processando...")
            for child_db_origin in child_origins_db:
                # Chamada recursiva, passando o limite de filhos
                child_node = _build_detailed_origin_node(
                    child_db_origin, run_id, db, max_depth, limit_children_per_node, current_depth + 1
                )
                origin_response_data["children"].append(child_node)
        else:
            logging.debug(f"Origin {db_origin.origin_id} não tem filhos ou todos foram filtrados pelo limite.")
    else:
        logging.info(
            f"Profundidade máxima ({max_depth}) atingida para Origin ID: {db_origin.origin_id}. Não buscando mais filhos.")

    # Retornar o objeto Pydantic construído a partir do dicionário
    return schemas.OriginWithUCsAndRelationshipsResponse(**origin_response_data)


@router.get(
    "/runs/{run_id}/origins/tree",
    response_model=List[schemas.OriginWithUCsAndRelationshipsResponse],
    summary="List ROOT Knowledge Unit Origins as a tree with UCs, Relations, children (recursive, with limits)"
)
def list_root_origins_as_tree(
        run_id: str,
        skip_roots: int = Query(0, ge=0, description="Number of root origins to skip (for pagination)"),
        limit_roots: int = Query(10, ge=1, le=100, description="Max number of root origins to return"),
        max_depth: int = Query(3, ge=0, le=10, description="Max depth of the children hierarchy (0 for only roots)"),
        limit_children_per_node: Optional[int] = Query(5, ge=1,
                                                       description="Max children per origin node. Omit or 0 for no limit (can be slow)."),
        # Ajustado o default e descrição
        origin_type_filter: Optional[str] = Query(None, alias="origin_type", description="Filter root origins by type"),
        db: Session = Depends(get_db)
):
    global uc_model_cache_for_request  # Limpa/inicializa o cache para esta nova requisição
    uc_model_cache_for_request = {}

    # Se limit_children_per_node for 0, tratar como None (sem limite)
    actual_limit_children = limit_children_per_node if limit_children_per_node and limit_children_per_node > 0 else None

    logging.info(
        f"Iniciando list_root_origins_as_tree para run_id: {run_id}, skip_roots: {skip_roots}, limit_roots: {limit_roots}, max_depth: {max_depth}, limit_children: {actual_limit_children}")

    run = crud.pipeline_run.get_run(db, run_id=run_id)
    if not run:
        logging.error(f"Pipeline run '{run_id}' não encontrado.")
        raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")

    # 1. Buscar as Origins RAIZ para o run_id com skip e limit
    query_root_origins = db.query(models.KnowledgeUnitOrigin).filter(
        models.KnowledgeUnitOrigin.pipeline_run_id == run_id,
        models.KnowledgeUnitOrigin.parent_community_id_of_origin == None  # Condição para ser raiz
    )
    if origin_type_filter:
        query_root_origins = query_root_origins.filter(models.KnowledgeUnitOrigin.origin_type == origin_type_filter)

    db_root_origins = query_root_origins.order_by(models.KnowledgeUnitOrigin.title).offset(skip_roots).limit(
        limit_roots).all()

    if not db_root_origins:
        logging.info(f"Nenhuma origin raiz encontrada para run_id: {run_id} com os filtros aplicados.")
        return []

    logging.info(f"Encontradas {len(db_root_origins)} origins raiz. Construindo árvore...")
    response_tree_list: List[schemas.OriginWithUCsAndRelationshipsResponse] = []

    for db_root_origin in db_root_origins:
        # Chama a função auxiliar para construir a árvore para cada raiz
        root_node = _build_detailed_origin_node(
            db_root_origin,
            run_id,
            db,
            max_depth,
            actual_limit_children,  # Passa o limite de filhos ajustado
            current_depth=0
        )
        response_tree_list.append(root_node)

    logging.info(f"Construção da árvore concluída para {len(response_tree_list)} raízes.")
    return response_tree_list