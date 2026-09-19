import logging
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from doc_etl_api.bootstrap import BootstrapState, BootstrapStatus, ingest_corpus
from doc_etl_api.config import Settings, settings
from doc_etl_api.jobs import JobRegistry
from doc_etl_api.pipeline import DoclingConverter, create_pipeline
from doc_etl_api.routes import router

logger = logging.getLogger(__name__)


def _fix_file_upload_schema(schema: dict) -> None:
    """Ensure Swagger UI renders the file picker for /sources/files."""
    component_name = "Body_ingest_files_sources_files_post"
    body_schema = schema.get("components", {}).get("schemas", {}).get(component_name)
    if body_schema is None:
        return
    files_items = body_schema.get("properties", {}).get("files", {}).get("items")
    if files_items is not None:
        files_items["type"] = "string"
        files_items["format"] = "binary"
        files_items.pop("contentMediaType", None)


def _configure_logging(app_settings: Settings) -> None:
    """Emit application logs at the configured level.

    Nothing else configures the root logger, so without this it sits at WARNING
    and every ``logger.info`` in the app -- search timings, bootstrap progress --
    is silently dropped. ``basicConfig`` is a no-op once the root logger already
    has handlers, so repeated app creation does not stack handlers.
    """
    logging.basicConfig(
        level=app_settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _load_models() -> tuple[DoclingConverter, HuggingFaceEmbedding]:
    logger.info("Loading Docling conversion models...")
    converter = DoclingConverter()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    embedding_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
    logger.info("Models loaded successfully.")
    return converter, embedding_model


def _start_corpus_bootstrap(app: FastAPI, app_settings: Settings) -> None:
    """Ingest the configured corpus in the background.

    Startup returns immediately: the corpus is ingested on a worker thread so
    boot time does not scale with Docling parse time, and the service answers
    requests (and reports its bootstrap state) while the corpus loads.
    """
    state: BootstrapState = app.state.bootstrap
    if app_settings.knowledge_corpus_path is None and not app_settings.knowledge_corpus_url_list:
        state.status = BootstrapStatus.DISABLED
        return

    state.status = BootstrapStatus.PENDING
    threading.Thread(
        target=ingest_corpus,
        args=(app.state.pipeline, app_settings, state),
        name="knowledge-bootstrap",
        daemon=True,
    ).start()


def create_app() -> FastAPI:
    _configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        description="Documents ETL API using Docling and LlamaIndex",
        version="0.1.0",
    )

    converter, embedding_model = _load_models()
    app.state.pipeline = create_pipeline(converter=converter, embedding_model=embedding_model)
    app.state.jobs = JobRegistry()
    app.state.bootstrap = BootstrapState()

    app.include_router(router)

    _start_corpus_bootstrap(app, settings)

    def custom_openapi() -> dict:
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )
        _fix_file_upload_schema(schema)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
    return app


app = create_app()


def run() -> None:
    uvicorn.run(
        "doc_etl_api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )


if __name__ == "__main__":
    run()
