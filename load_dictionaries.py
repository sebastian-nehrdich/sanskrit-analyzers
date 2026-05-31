import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Tuple

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError as ESConnectionError, ApiError


DATA_DIR = Path(os.getenv("DICTIONARY_DATA_DIR", "data"))
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("ES_DICTIONARY_INDEX", "sanskrit_dictionaries")
BULK_CHUNK_SIZE = int(os.getenv("ES_BULK_CHUNK_SIZE", "1000"))
ES_WAIT_SECONDS = int(os.getenv("ES_WAIT_SECONDS", "60"))


INDEX_BODY = {
    "settings": {
        "analysis": {
            "analyzer": {
                "edge_ngram_analyzer": {
                    "type": "custom",
                    "tokenizer": "edge_ngram_tokenizer",
                    "filter": ["lowercase"],
                }
            },
            "tokenizer": {
                "edge_ngram_tokenizer": {
                    "type": "edge_ngram",
                    "min_gram": 1,
                    "max_gram": 20,
                    "token_chars": ["letter", "digit"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "key": {
                "type": "text",
                "analyzer": "edge_ngram_analyzer",
                "search_analyzer": "standard",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "source": {"type": "keyword"},
            "entry_html": {"type": "text"},
            "entry_txt": {"type": "text"},
            "entry_md": {"type": "text"},
            "page_link": {"type": "keyword"}
        }
    }
}


def wait_for_elasticsearch(es: Elasticsearch, timeout: int) -> None:
    """Wait until Elasticsearch is reachable or raise a timeout error."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if es.ping():
                return
        except ESConnectionError:
            pass
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Elasticsearch at {ELASTICSEARCH_URL} did not become ready within {timeout} seconds.")


def ensure_index(es: Elasticsearch) -> None:
    """Create the dictionary index with the desired mapping if it does not exist."""
    try:
        if es.indices.exists(index=INDEX_NAME):
            return
        es.indices.create(index=INDEX_NAME, body=INDEX_BODY)
    except ApiError as exc:
        if exc.error == "resource_already_exists_exception":
            return
        raise


def ndjson_files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.glob("*.ndjson")):
        if path.is_file():
            yield path


def iter_documents(files: Iterable[Path]) -> Iterator[Tuple[str, dict]]:
    for file_path in files:
        source_name = file_path.stem
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as err:
                    print(f"Skipping malformed JSON in {file_path} at line {line_number}: {err}", file=sys.stderr)
                    continue

                key_value = payload.get("key") or payload.get("keyword")
                if not key_value:
                    print(f"Skipping entry without key in {file_path} line {line_number}", file=sys.stderr)
                    continue

                document = dict(payload)
                document["key"] = key_value
                document.setdefault("source", source_name)

                doc_id = f"{source_name}:{line_number}:{key_value}".replace(" ", "_")
                yield doc_id, document


def bulk_actions(documents: Iterable[Tuple[str, dict]]) -> Iterator[dict]:
    for doc_id, document in documents:
        yield {
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": doc_id,
            "_source": document,
        }


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dictionary data directory not found: {DATA_DIR}")

    es = Elasticsearch(ELASTICSEARCH_URL, request_timeout=30)
    print(f"Waiting for Elasticsearch at {ELASTICSEARCH_URL}...")
    wait_for_elasticsearch(es, ES_WAIT_SECONDS)

    print(f"Ensuring index '{INDEX_NAME}' exists with fuzzy-friendly mapping...")
    ensure_index(es)

    files = list(ndjson_files(DATA_DIR))
    if not files:
        print(f"No .ndjson files found in {DATA_DIR}; nothing to load.")
        return

    total_docs = 0
    for success, response in helpers.streaming_bulk(
        es,
        bulk_actions(iter_documents(files)),
        chunk_size=BULK_CHUNK_SIZE,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if success:
            total_docs += 1
        else:
            action, result = response.popitem()
            error = result.get("error", {})
            print(f"Failed to index document during {action}: {error}", file=sys.stderr)

    print(f"Indexed approximately {total_docs} dictionary documents into '{INDEX_NAME}'.")
    es.indices.refresh(index=INDEX_NAME)


if __name__ == "__main__":
    main()
