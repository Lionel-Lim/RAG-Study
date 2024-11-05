# Prerequisites:
# ! pip install llama-index llama-index-vector-stores-vertexaivectorsearch llama-index-llms-vertex llama-index-storage-docstore-firestore
# ! pip install --upgrade google-cloud-documentai
# REFERENCE: https://cloud.google.com/vertex-ai/docs/vector-search/quickstart#enable-apis
# ! gcloud services enable compute.googleapis.com aiplatform.googleapis.com storage.googleapis.com --project ai-sandbox-company-73

# Import the required libraries
import logging
import os
import sys
import yaml
from google.cloud import aiplatform
from docai_parser import DocAIParser
from google.oauth2 import service_account

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from llama_index.storage.docstore.firestore import FirestoreDocumentStore
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex

# Add the common directory to the system path
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
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
    config_path = os.path.join(os.path.dirname(""), "..", "..", "common", "config.yaml")
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
google_credential_path = os.path.join(
    os.path.dirname(""), "..", "..", GOOGLE_CREDENTIAL_PATH
)
google_credential = service_account.Credentials.from_service_account_file(
    google_credential_path
)


class QuesionsAnswered(BaseModel):
    """List of Questions Answered by Document"""

    questions_list: list[str]


def create_qa_index(li_docs, docstore, embed_model, llm):
    """
    Creates index of hypothetical questions

    Args:
        li_docs: List of documents
        docstore: Firestore docstore
        embed_model: Vertex Text Embedding
        llm: Vertex LLM

    Returns:

    """
    qa_index, qa_endpoint = get_or_create_existing_index(
        QA_INDEX_NAME, QA_ENDPOINT_NAME, APPROXIMATE_NEIGHBORS_COUNT
    )
    qa_vector_store = VertexAIVectorStore(
        project_id=PROJECT_ID,
        region=LOCATION,
        index_id=qa_index.name,
        endpoint_id=qa_endpoint.name,
        gcs_bucket_name=DOCSTORE_BUCKET_NAME,
    )
    qa_extractor = QuestionsAnsweredExtractor(
        llm, questions=5, prompt_template=QA_EXTRACTION_PROMPT
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
    logger.info("downloading data")
    download_bucket_with_transfer_manager(
        INPUT_BUCKET_NAME, prefix=BUCKET_PREFIX, destination_directory=local_data_path
    )

    # Parse documents using Document AI
    try:
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
