import logging

from backend.app.dependencies import get_index_manager, get_prompts
from backend.app.models import RAGRequest
from datasets import Dataset
from fastapi import APIRouter, Depends
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
import pandas as pd
from ragas import evaluate

# TODO: context_relevancy is now removed from ragas.metrics (https://github.com/explodinggradients/ragas/issues/1210)
# from ragas.metrics import answer_relevancy, context_relevancy, faithfulness

from ragas.metrics import answer_relevancy, faithfulness
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core import StorageContext, VectorStoreIndex
from google.cloud import aiplatform, firestore, firestore_admin_v1
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from llama_index.storage.docstore.firestore import FirestoreDocumentStore
from llama_index.embeddings.vertex import VertexTextEmbedding

router = APIRouter()
logger = logging.getLogger(__name__)


# Create simple retriever endpoint using LlamaIndex AutoMergingRetriever
@router.post("/query_rag_test")
async def auto_merging_retriever(
    rag_request: RAGRequest,
    index_manager=Depends(get_index_manager),
    prompts=Depends(get_prompts),
) -> dict:
    query_engine = index_manager.get_query_engine(
        prompts=prompts,
        llm_name=rag_request.llm_name,
        temperature=rag_request.temperature,
        similarity_top_k=rag_request.similarity_top_k,
        retrieval_strategy=rag_request.retrieval_strategy,
        use_hyde=rag_request.use_hyde,
        use_refine=rag_request.use_refine,
        use_node_rerank=rag_request.use_node_rerank,
        qa_followup=rag_request.qa_followup,
        hybrid_retrieval=rag_request.hybrid_retrieval,
    )
    # Instantiating the index at retrieval time:
    aiplatform.init(project=index_manager.project_id, location=index_manager.location)
    # Get the Vector Search index
    indexes = aiplatform.MatchingEngineIndex.list(
        filter=f'display_name="{index_manager.base_index_name}"'
    )
    if not indexes:
        raise ValueError(
            f"No index found with display name: {index_manager.base_index_name}"
        )
    vs_index = indexes[0]
    # Get the Vector Search endpoint
    endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f'display_name="{index_manager.base_endpoint_name}"'
    )
    if not endpoints:
        raise ValueError(
            f"No endpoint found with display name: {index_manager.base_endpoint_name}"
        )
    vs_endpoint = endpoints[0]

    # Create the vector store
    vector_store = VertexAIVectorStore(
        project_id=index_manager.project_id,
        region=index_manager.location,
        index_id=vs_index.resource_name.split("/")[-1],
        endpoint_id=vs_endpoint.resource_name.split("/")[-1],
        gcs_bucket_name=index_manager.vs_bucket_name,
    )
    if index_manager.firestore_db_name and index_manager.firestore_namespace:
        docstore = FirestoreDocumentStore.from_database(
            project=index_manager.project_id,
            database=index_manager.firestore_db_name,
            namespace=index_manager.firestore_namespace,
        )
    else:
        docstore = None

    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, docstore=docstore
    )
    # Create and return the index
    embed_model = VertexTextEmbedding(
        model_name=index_manager.embeddings_model_name,
        project=index_manager.project_id,
        location=index_manager.location,
        credentials=index_manager.credential,
    )
    vector_store_index = VectorStoreIndex(
        nodes=[], storage_context=storage_context, embed_model=embed_model
    )
    baseline_retriever = vector_store_index.as_retriever()
    nodes_with_scores = baseline_retriever.retrieve(rag_request.query)
    return {"response": nodes_with_scores}


@router.post("/query_rag")
async def query_rag(
    rag_request: RAGRequest,
    index_manager=Depends(get_index_manager),
    prompts=Depends(get_prompts),
) -> dict:
    query_engine = index_manager.get_query_engine(
        prompts=prompts,
        llm_name=rag_request.llm_name,
        temperature=rag_request.temperature,
        similarity_top_k=rag_request.similarity_top_k,
        retrieval_strategy=rag_request.retrieval_strategy,
        use_hyde=rag_request.use_hyde,
        use_refine=rag_request.use_refine,
        use_node_rerank=rag_request.use_node_rerank,
        qa_followup=rag_request.qa_followup,
        hybrid_retrieval=rag_request.hybrid_retrieval,
    )
    if rag_request.use_react:
        react_agent = index_manager.get_react_agent(
            prompts=prompts,
            llm_name=rag_request.llm_name,
            temperature=rag_request.temperature,
        )
        response = await react_agent.achat(rag_request.query)
    else:
        response = await query_engine.aquery(rag_request.query)

    if rag_request.evaluate_response:
        retrieved_contexts = [r.node.text for r in response.source_nodes]
        eval_df = pd.DataFrame(
            {
                "question": rag_request.query,
                "answer": [response.response],
                "contexts": [retrieved_contexts],
            }
        )
        eval_df_ds = Dataset.from_pandas(eval_df)

        vertexai_llm = ChatVertexAI(model_name=rag_request.eval_model_name)
        vertexai_embeddings = VertexAIEmbeddings(
            model_name=rag_request.embedding_model_name
        )

        # metrics = [answer_relevancy, faithfulness, context_relevancy]
        metrics = [answer_relevancy, faithfulness]
        result = evaluate(
            eval_df_ds,
            metrics=metrics,
            llm=vertexai_llm,
            embeddings=vertexai_embeddings,
        )
        result_dict = (
            result.to_pandas()[
                # ["answer_relevancy", "faithfulness", "context_relevancy"]
                ["answer_relevancy", "faithfulness"]
            ]
            .fillna(0)
            .iloc[0]
            .to_dict()
        )
        retrieved_context_dict = {"retrieved_chunks": response.source_nodes}
        logger.info(result_dict)
        return {"response": response.response} | result_dict | retrieved_context_dict
    else:
        return {"response": response.response}
