# Prerequisites:
# ! pip install llama-index llama-index-vector-stores-vertexaivectorsearch llama-index-llms-vertex llama-index-storage-docstore-firestore
# ! pip install google-cloud-documentai pydantic
# REFERENCE: https://cloud.google.com/vertex-ai/docs/vector-search/quickstart#enable-apis
# ! gcloud services enable compute.googleapis.com aiplatform.googleapis.com storage.googleapis.com --project ai-sandbox-company-73

# Import the required libraries
import logging
import os
import sys
import yaml
import asyncio

from google.cloud import aiplatform  # Vertex AI Platform
from google.oauth2 import service_account  # Service Account Credentials
from google.api_core.exceptions import NotFound
from pydantic import BaseModel  # Pydantic is a data validation and parsing library
from tqdm.asyncio import (
    tqdm_asyncio,
)  # tqdm is a library that allows you to display progress bars while running code

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
    get_leaf_nodes,
)
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from llama_index.storage.docstore.firestore import FirestoreDocumentStore
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex

# Add the common directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from docai_parser import DocAIParser  # Document AI Parser
from vector_search_utils import get_or_create_existing_index
from backend.indexing.prompts import QA_EXTRACTION_PROMPT, QA_PARSER_PROMPT
from common.utils import (
    create_pdf_blob_list,
    download_bucket_with_transfer_manager,
    link_nodes,
)

logging.basicConfig(level=logging.INFO)  # Set the desired logging level
logger = logging.getLogger(__name__)


# Load configuration from config.yaml
def load_config():
    config_path = os.path.join(os.path.dirname(""), "common", "config.yaml")
    print(config_path)
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


# Load configuration
config = load_config()

# Initialize parameters
PROJECT_ID = config["project_id"]
LOCATION = config["location"]
INPUT_BUCKET_NAME = config["input_bucket_name"]
DOCSTORE_BUCKET_NAME = config["docstore_bucket_name"]
INDEX_ID = config["index_id"]
VECTOR_INDEX_NAME = config["vector_index_name"]
INDEX_ENDPOINT_NAME = config["index_endpoint_name"]
INDEXING_METHOD = config["indexing_method"]
CHUNK_SIZES = config["chunk_sizes"]
EMBEDDINGS_MODEL_NAME = config["embeddings_model_name"]
LLM_MODEL_NAME = config["llm_model_name"]
APPROXIMATE_NEIGHBORS_COUNT = config["approximate_neighbors_count"]
BUCKET_PREFIX = config["bucket_prefix"]
VECTOR_DATA_PREFIX = config["vector_data_prefix"]
CHUNK_SIZE = config.get("chunk_size", 512)
CHUNK_OVERLAP = config.get("chunk_overlap", 50)
DOCAI_LOCATION = config["docai_location"]
DOCAI_PROCESSOR_DISPLAY_NAME = config["document_ai_processor_display_name"]
DOCAI_PROCESSOR_ID = config.get("docai_processor_id")
CREATE_DOCAI_PROCESSOR = config.get("create_docai_processor", False)
FIRESTORE_DB_NAME = config.get("firestore_db_name")
FIRESTORE_NAMESPACE = config.get("firestore_namespace")
QA_INDEX_NAME = config.get("qa_index_name")
QA_ENDPOINT_NAME = config.get("qa_endpoint_name")
GCS_OUTPUT_PATH = f"gs://{DOCSTORE_BUCKET_NAME}/{VECTOR_DATA_PREFIX}/docai_output/"
GOOGLE_CREDENTIAL_PATH = config.get("credential")

# Google Service Account credentials
google_credential_path = os.path.join(os.path.dirname(""), GOOGLE_CREDENTIAL_PATH)
google_credential = service_account.Credentials.from_service_account_file(
    google_credential_path
)


class QuesionsAnswered(BaseModel):
    """List of Questions Answered by Document"""

    questions_list: list[str]


def create_qa_index(
    li_docs: list[Document],
    docstore: FirestoreDocumentStore,
    embed_model: VertexTextEmbedding,
    llm: Vertex,
):
    """Create a QA index for the parsed documents.

    Args:
        li_docs (list[Document]): List of parsed documents
        docstore (FirestoreDocumentStore): Firestore Document Store
        embed_model (VertexTextEmbedding): Vertex Text Embedding model
        llm (Vertex): Vertex model

    Returns:
        None
    """
    qa_index, qa_endpoint = get_or_create_existing_index(
        QA_INDEX_NAME, QA_ENDPOINT_NAME, APPROXIMATE_NEIGHBORS_COUNT, google_credential
    )
    qa_vector_store = VertexAIVectorStore(
        project_id=PROJECT_ID,
        region=LOCATION,
        index_id=qa_index.name,
        endpoint_id=qa_endpoint.name,
        gcs_bucket_name=DOCSTORE_BUCKET_NAME,
        credentials_path=google_credential_path,
    )
    qa_extractor = QuestionsAnsweredExtractor(
        llm, questions=5, prompt_template=QA_EXTRACTION_PROMPT
    )

    async def extract_batch(li_docs):
        return await tqdm_asyncio.gather(
            *[qa_extractor._aextract_questions_from_node(doc) for doc in li_docs]
        )

    loop = asyncio.get_event_loop()
    metadata_list = loop.run_until_complete(extract_batch(li_docs))

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=QuesionsAnswered,
        prompt_template_str=QA_PARSER_PROMPT,
        verbose=True,
    )

    async def parse_batch(metadata_list):
        return await asyncio.gather(
            *[program.acall(questions_list=x) for x in metadata_list],
            return_exceptions=True,
        )

    parsed_questions = loop.run_until_complete(parse_batch(metadata_list))

    loop.close()

    q_docs = []
    for doc, questions in zip(li_docs, parsed_questions):
        if isinstance(questions, Exception):
            logger.infor(f"Unparsable questions exception: {questions}")
            continue
        else:
            for q in questions.questions_list:
                logger.info(f"Question extracted: {q}")
                q_doc = Document(text=q)
                q_doc.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=doc.doc_id
                )
                q_docs.append(q_doc)
    docstore.add_documents(li_docs)
    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=qa_vector_store
    )
    VectorStoreIndex(
        nodes=q_docs,
        storage_context=storage_context,
        embed_model=embed_model,
        llm=llm,
    )


def create_hierarchical_index(
    li_docs: list[Document],
    docstore: FirestoreDocumentStore,
    vector_store: VertexAIVectorStore,
    embed_model: VertexTextEmbedding,
    llm: Vertex,
):
    """Create a hierarchical index for the parsed documents.

    Args:
        li_docs (list[Document]): List of parsed documents
        docstore (FirestoreDocumentStore): Firestore Document Store
        vector_store (VertexAIVectorStore): Vertex AI Vector Store
        embed_model (VertexTextEmbedding): Vertex Text Embedding model
        llm (Vertex): Vertex model

    Returns:
        None
    """
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)
    nodes = node_parser.get_nodes_from_documents(li_docs)

    # leaf_nodes = node_parser.get_leaf_nodes(nodes)
    leaf_nodes = get_leaf_nodes(nodes)
    num_leaf_nodes = len(leaf_nodes)
    num_nodes = len(nodes)
    logger.info(f"There are {num_nodes} nodes and {num_leaf_nodes} leaf nodes.")
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store
    )
    VectorStoreIndex(
        nodes=leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        llm=llm,
    )


def create_flat_index(li_docs, docstore, vector_store, embed_model, llm):
    sentence_splitter = SentenceSplitter(chunk_size=CHUNK_OVERLAP)
    # Chunk into granular chunks manually
    node_chunk_list = []
    for doc in li_docs:
        doc_dict = doc.to_dict()
        metadata = doc_dict.pop("metadata")
        doc_dict.update(metadata)
        chunks = sentence_splitter.get_nodes_from_documents([doc])

        # Create nodes with relationships and flatten
        nodes = []
        for chunk in chunks:
            text = chunk.pop("text")
            doc_source_id = doc.doc_id
            node = TextNode(text=text, metadata=chunk)
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=doc_source_id
            )
            nodes.append(node)

        nodes = link_nodes(nodes)
        node_chunk_list.extend(nodes)

    nodes = node_chunk_list
    logger.info("embedding...")
    docstore.add_documents(li_docs)
    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store
    )

    for node in nodes:
        node.metadata.pop("excluded_embed_metadata_keys", None)
        node.metadata.pop("excluded_llm_metadata_keys", None)

    # Creating an index automatically embeds and creates the
    # vector db collection
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        llm=llm,
    )


def main():
    # Initialize Vertex AI and create index and endpoint
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Creating Vector Search Index
    vs_index, vs_endpoint = get_or_create_existing_index(
        VECTOR_INDEX_NAME,
        INDEX_ENDPOINT_NAME,
        APPROXIMATE_NEIGHBORS_COUNT,
        google_credential,
    )

    # Vertex AI Vector Search Vector DB and Firestore Docstore
    vector_store = VertexAIVectorStore(
        project_id=PROJECT_ID,
        region=LOCATION,
        index_id=vs_index.name,  # Use .name instead of .resource_name as it contains the full path
        endpoint_id=vs_endpoint.name,  # Use .name instead of .resource_name
        gcs_bucket_name=DOCSTORE_BUCKET_NAME,
        credentials_path=google_credential_path,
    )
    # TODO: Add service account credentials for Firestore
    docstore = FirestoreDocumentStore.from_database(
        project=PROJECT_ID, database=FIRESTORE_DB_NAME, namespace=FIRESTORE_NAMESPACE
    )
    try:
        docstore.get_document("test")
    except NotFound:
        # If the database does exist, it will throw a ValueError.
        logger.info(
            "Firestore document store not found. Creating Firestore document store."
        )
        return None
    except ValueError:
        logger.info("Firestore DB is found.")

    # Setup embedding model and LLM
    embed_model = VertexTextEmbedding(
        model_name=EMBEDDINGS_MODEL_NAME,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=google_credential,
    )
    llm = Vertex(model=LLM_MODEL_NAME, temperature=0.0)
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Initialise Document AI parser
    parser = DocAIParser(
        project_id=PROJECT_ID,
        location=DOCAI_LOCATION,
        processor_name=f"projects/{PROJECT_ID}/locations/{DOCAI_LOCATION}/processors/{DOCAI_PROCESSOR_ID}",  # noqa: E501
        gcs_output_path=GCS_OUTPUT_PATH,
    )

    # Download data from specific bucket and parse
    local_data_path = os.path.join("/tmp", BUCKET_PREFIX)
    os.makedirs(local_data_path, exist_ok=True)
    blobs = create_pdf_blob_list(INPUT_BUCKET_NAME, BUCKET_PREFIX)
    # DEBUG: Select only the first 5 blobs for testing
    blobs = blobs[:5]
    logger.info("downloading data")
    download_bucket_with_transfer_manager(
        INPUT_BUCKET_NAME, prefix=BUCKET_PREFIX, destination_directory=local_data_path
    )

    # Parse documents using Document AI
    try:
        # Log the number of documents parsed and starting time using logger
        logger.info(f"Number of documents to parse: {len(blobs)}")
        logger.info("Parsing documents...")
        parsed_docs, raw_results = parser.batch_parse(
            blobs, chunk_size=CHUNK_SIZE, include_ancestor_headings=True
        )
        print(f"Number of documents parsed by Document AI: {len(parsed_docs)}")
        if parsed_docs:
            print(
                f"First parsed document text (first 100 chars): {parsed_docs[0].text[:100]}..."
            )
        else:
            print("No documents were parsed by Document AI")

        # Print raw results for debugging
        print("Raw results from Document AI:")
        for result in raw_results:
            print(f"  Source: {result.source_path}")
            print(f"  Parsed: {result.parsed_path}")

    except Exception as e:
        print(f"Error parsing documents: {str(e)}")
        parsed_docs = []
        raw_results = []

    # Turn each parsed document into llama_index Document
    li_docs = [Document(text=doc.text, metadata=doc.metadata) for doc in parsed_docs]

    if QA_INDEX_NAME or QA_ENDPOINT_NAME:
        create_qa_index(li_docs, docstore, embed_model, llm)

    if INDEXING_METHOD == "hierarchical":
        create_hierarchical_index(li_docs, docstore, vector_store, embed_model, llm)

    elif INDEXING_METHOD == "flat":
        create_flat_index(li_docs, docstore, vector_store, embed_model, llm)


if __name__ == "__main__":
    main()
