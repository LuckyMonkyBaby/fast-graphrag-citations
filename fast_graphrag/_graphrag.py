"""This module implements a Graph-based Retrieval-Augmented Generation (GraphRAG) system."""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

from fast_graphrag._llm import BaseLLMService, format_and_send_prompt
from fast_graphrag._llm._base import BaseEmbeddingService
from fast_graphrag._models import TAnswer
from fast_graphrag._policies._base import BaseEdgeUpsertPolicy, BaseGraphUpsertPolicy, BaseNodeUpsertPolicy
from fast_graphrag._prompt import PROMPTS
from fast_graphrag._services._chunk_extraction import BaseChunkingService
from fast_graphrag._services._information_extraction import BaseInformationExtractionService
from fast_graphrag._services._state_manager import BaseStateManagerService
from fast_graphrag._storage._base import BaseGraphStorage, BaseIndexedKeyValueStorage, BaseVectorStorage
from fast_graphrag._types import GTChunk, GTEdge, GTEmbedding, GTHash, GTId, GTNode, TContext, TDocument, TQueryResponse
from fast_graphrag._utils import TOKEN_TO_CHAR_RATIO, get_event_loop, logger


@dataclass
class InsertParam:
    pass


@dataclass
class QueryParam:
    # Prompt type constants
    PROMPT_DEFAULT = "DEFAULT"
    PROMPT_WITH_REFERENCES = "WITH_REFERENCES"
    PROMPT_JSON = "JSON"
    PROMPT_SCHEMA_JSON = "SCHEMA_JSON"
    PROMPT_TABLE = "TABLE"
    PROMPT_BULLET_POINTS = "BULLET_POINTS"
    PROMPT_CUSTOM = "CUSTOM"
    
    with_references: bool = field(default=False)
    only_context: bool = field(default=False)
    entities_max_tokens: int = field(default=4000)
    relations_max_tokens: int = field(default=3000)
    chunks_max_tokens: int = field(default=9000)
    prompt_type: str = field(default=PROMPT_DEFAULT)
    custom_prompt_key: Optional[str] = field(default=None)
    include_sub_chunks: bool = field(default=True) 


@dataclass
class BaseGraphRAG(Generic[GTEmbedding, GTHash, GTChunk, GTNode, GTEdge, GTId]):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""

    working_dir: str = field()
    domain: str = field()
    example_queries: str = field()
    entity_types: List[str] = field()
    n_checkpoints: int = field(default=0)

    llm_service: BaseLLMService = field(init=False, default_factory=lambda: BaseLLMService())
    chunking_service: BaseChunkingService[GTChunk] = field(init=False, default_factory=lambda: BaseChunkingService())
    information_extraction_service: BaseInformationExtractionService[GTChunk, GTNode, GTEdge, GTId] = field(
        init=False,
        default_factory=lambda: BaseInformationExtractionService(
            graph_upsert=BaseGraphUpsertPolicy(
                config=None,
                nodes_upsert_cls=BaseNodeUpsertPolicy,
                edges_upsert_cls=BaseEdgeUpsertPolicy,
            )
        ),
    )
    state_manager: BaseStateManagerService[GTNode, GTEdge, GTHash, GTChunk, GTId, GTEmbedding] = field(
        init=False,
        default_factory=lambda: BaseStateManagerService(
            workspace=None,
            graph_storage=BaseGraphStorage[GTNode, GTEdge, GTId](config=None),
            entity_storage=BaseVectorStorage[GTId, GTEmbedding](config=None),
            chunk_storage=BaseIndexedKeyValueStorage[GTHash, GTChunk](config=None),
            embedding_service=BaseEmbeddingService(),
            node_upsert_policy=BaseNodeUpsertPolicy(config=None),
            edge_upsert_policy=BaseEdgeUpsertPolicy(config=None),
        ),
    )

    def insert(
        self,
        content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True
    ) -> Tuple[int, int, int]:
        return get_event_loop().run_until_complete(self.async_insert(content, metadata, params, show_progress))

    async def async_insert(
        self,
        content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True
    ) -> Tuple[int, int, int]:
        """Insert a new memory or memories into the graph.

        Args:
            content (str | list[str]): The data to be inserted. Can be a single string or a list of strings.
            metadata (dict, optional): Additional metadata associated with the data. Defaults to None.
            params (InsertParam, optional): Additional parameters for the insertion. Defaults to None.
            show_progress (bool, optional): Whether to show the progress bar. Defaults to True.
        """
        if params is None:
            params = InsertParam()

        if isinstance(content, str):
            content = [content]
        if isinstance(metadata, dict):
            metadata = [metadata]

        if metadata is None or isinstance(metadata, dict):
            data = (TDocument(data=c, metadata=metadata or {}) for c in content)
        else:
            data = (TDocument(data=c, metadata=m or {}) for c, m in zip(content, metadata))

        try:
            await self.state_manager.insert_start()
            # Chunk the data
            chunked_documents = await self.chunking_service.extract(data=data)

            # Filter the chunks checking for duplicates
            new_chunks_per_data = await self.state_manager.filter_new_chunks(chunks_per_data=chunked_documents)

            # Extract entities and relationships from the new chunks only
            subgraphs = self.information_extraction_service.extract(
                llm=self.llm_service,
                documents=new_chunks_per_data,
                prompt_kwargs={
                    "domain": self.domain,
                    "example_queries": self.example_queries,
                    "entity_types": ",".join(self.entity_types),
                },
                entity_types=self.entity_types,
            )
            if len(subgraphs) == 0:
                logger.info("No new entities or relationships extracted from the data.")

            # Update the graph with the new entities, relationships, and chunks
            await self.state_manager.upsert(
                llm=self.llm_service, subgraphs=subgraphs, documents=new_chunks_per_data, show_progress=show_progress
            )

            # Commit the changes if all is successful
            await self.state_manager.insert_done()

            # Return the total number of entities, relationships, and chunks
            return (
                await self.state_manager.get_num_entities(),
                await self.state_manager.get_num_relations(),
                await self.state_manager.get_num_chunks(),
            )
        except Exception as e:
            logger.error(f"Error during insertion: {e}")
            raise e

    def query(self, query: str, params: Optional[QueryParam] = None) -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
        async def _query() -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
            await self.state_manager.query_start()
            try:
                answer = await self.async_query(query, params)
                return answer
            except Exception as e:
                logger.error(f"Error during query: {e}")
                raise e
            finally:
                await self.state_manager.query_done()

        return get_event_loop().run_until_complete(_query())

    async def async_query(
        self, query: Optional[str], params: Optional[QueryParam] = None
    ) -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
        """Query the graph with a given input.

        Args:
            query (str): The query string to search for in the graph.
            params (QueryParam, optional): Additional parameters for the query. Defaults to None.

        Returns:
            TQueryResponse: The result of the query (response + context).
        """
        if query is None or len(query) == 0:
            return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](
                response=PROMPTS["fail_response"], context=TContext([], [], [])
            )
        
        if params is None:
            params = QueryParam()

        # Extract entities from query
        extracted_entities = await self.information_extraction_service.extract_entities_from_query(
            llm=self.llm_service, query=query, prompt_kwargs={}
        )

        # Retrieve relevant state
        context = await self.state_manager.get_context(query=query, entities=extracted_entities)
        if context is None:
            return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](
                response=PROMPTS["fail_response"], context=TContext([], [], [])
            )

        # Get context string with appropriate truncation
        context_str = context.truncate(
            max_chars={
                "entities": params.entities_max_tokens * TOKEN_TO_CHAR_RATIO,
                "relations": params.relations_max_tokens * TOKEN_TO_CHAR_RATIO,
                "chunks": params.chunks_max_tokens * TOKEN_TO_CHAR_RATIO,
            },
            output_context_str=not params.only_context
        )
        
        # Handle only_context case
        if params.only_context:
            return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](response="", context=context)
        
        # Select the appropriate prompt based on prompt_type
        prompt_key = self._get_prompt_key(params)

        
        # Send the prompt to the LLM
        llm_response, _ = await format_and_send_prompt(
            prompt_key=prompt_key,
            llm=self.llm_service,
            format_kwargs={
                "query": query,
                "context": context_str
            },
            response_model=TAnswer,
        )
        answer = llm_response.answer

        return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](response=answer, context=context)
    
    def _get_prompt_key(self, params: QueryParam) -> str:
        """Determine which prompt key to use based on the query parameters."""
        # Handle legacy 'with_references' parameter for backward compatibility
        if params.prompt_type == QueryParam.PROMPT_DEFAULT and params.with_references:
            return "generate_response_query_with_references"
        
        # For custom prompts, use the provided key
        if params.prompt_type == QueryParam.PROMPT_CUSTOM:
            if params.custom_prompt_key is None:
                logger.warning("Custom prompt type selected but no custom_prompt_key provided. Using default prompt.")
                return "generate_response_query_no_references"
            return params.custom_prompt_key
        
        # For predefined prompt types
        prompt_keys = {
            QueryParam.PROMPT_DEFAULT: "generate_response_query_no_references",
            QueryParam.PROMPT_WITH_REFERENCES: "generate_response_query_with_references",
            QueryParam.PROMPT_JSON: "generate_json_response_query",
            QueryParam.PROMPT_SCHEMA_JSON: "generate_schema_json_response_query",
            QueryParam.PROMPT_TABLE: "generate_table_response_query",
            QueryParam.PROMPT_BULLET_POINTS: "generate_bullet_points_response_query",
        }
        
        # Return the appropriate prompt key, defaulting to the basic one if not found
        return prompt_keys.get(params.prompt_type, "generate_response_query_no_references")

    def save_graphml(self, output_path: str) -> None:
        """Save the graph in GraphML format."""
        async def _save_graphml() -> None:
            await self.state_manager.query_start()
            try:
                await self.state_manager.save_graphml(output_path)
            finally:
                await self.state_manager.query_done()
        get_event_loop().run_until_complete(_save_graphml())