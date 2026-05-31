import os
from difflib import SequenceMatcher

from fastapi import FastAPI, HTTPException
import transformers
import ctranslate2
from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions
from sanskrit_postprocessor import SanskritPostprocessor
from endpoint_models import (
    TaggingRequest,
    TaggingResponse,
    TaggerParsedRequest,
    TaggerParsedResponse,
    Sentence,
    Lemma,
    DictionaryEntry,
    DictionaryResponse,
)

# Assuming you have these imports and definitions
# from transformers import MarianTokenizer
# from ctranslate2 import Translator
# tokenizer = MarianTokenizer.from_pretrained(...)
# translator = Translator(...)
max_length = 512

ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_DICTIONARY_INDEX", "sanskrit_dictionaries")
_es_client: Elasticsearch | None = None


def _get_es_client() -> Elasticsearch:
    global _es_client
    if _es_client is None:
        _es_client = Elasticsearch(ES_URL)
    return _es_client


def _build_translator():
    device = os.getenv("CTRANSLATE_DEVICE", "cuda").lower()
    model_path = os.getenv("CTRANSLATE_MODEL_PATH", "ctranslate/ct2-byt5sanskrit-int8")
    kwargs = {"device": device}

    if device != "cpu":
        index_env = os.getenv("CTRANSLATE_DEVICE_INDEX", "0")
        indices = [int(item.strip()) for item in index_env.split(",") if item.strip()]
        if indices:
            kwargs["device_index"] = indices
    return ctranslate2.Translator(model_path, **kwargs)


translator = _build_translator()

tokenizer = transformers.AutoTokenizer.from_pretrained("chronbmm/sanskrit5-multitask")

root_path = os.getenv("FASTAPI_ROOT_PATH", "")
app = FastAPI(root_path=root_path or "")

# Initialize the Sanskrit postprocessor
postprocessor = SanskritPostprocessor()

PREFIX_MAP = {
    "unsandhied": "S ",
    "unsandhied-morphosyntax": "SM ",
    "lemma": "L ",
    "lemma-morphosyntax": "LM ",
    "unsandhied-lemma-morphosyntax": "SLM ",
}


@app.get("/dictionary/search", response_model=DictionaryResponse)
async def dictionary_search(query: str, min_similarity: float = 0.8, max_results: int = 25):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if not 0 <= min_similarity <= 1:
        raise HTTPException(status_code=400, detail="min_similarity must be within [0, 1].")
    if max_results <= 0:
        raise HTTPException(status_code=400, detail="max_results must be greater than zero.")

    es = _get_es_client()
    hits = []
    try:
        response = es.search(
            index=ES_INDEX,
            size=max_results * 5,
            query={
                "bool": {
                    "should": [
                        {"term": {"key.keyword": query}},
                        {"term": {"keyword": query}},
                        {
                            "match": {
                                "key": {
                                    "query": query,
                                    "fuzziness": "AUTO",
                                    "operator": "and"
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        )
        hits = response.get("hits", {}).get("hits", [])
    except es_exceptions.NotFoundError:
        raise HTTPException(status_code=404, detail=f"Dictionary index '{ES_INDEX}' not found.")
    except es_exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Dictionary service unavailable.")

    results: list[DictionaryEntry] = []

    for hit in hits:
        source = hit.get("_source", {})
        lemma = (source.get("key") or source.get("keyword") or "").strip()
        if not lemma:
            continue

        normalized_query = query.strip().lower()
        normalized_lemma = lemma.lower()
        if normalized_query == normalized_lemma:
            similarity = 1.0
        else:
            similarity = SequenceMatcher(None, normalized_query, normalized_lemma).ratio()
        if similarity < min_similarity:
            continue

        source_name = source.get("source")
        entry_txt = source.get("entry_txt")
        entry_md = source.get("entry_md")

        if similarity >= 0.99 and source_name and entry_txt:
            formatted_entry_txt = f"{source_name}: {entry_txt}"
        else:
            formatted_entry_txt = entry_txt

        if similarity >= 0.99 and source_name and entry_md:
            formatted_entry_md = f"{source_name}: {entry_md}"
        else:
            formatted_entry_md = entry_md

        results.append(
            DictionaryEntry(
                lemma=lemma,
                source=source_name,
                entry_html=source.get("entry_html"),
                entry_txt=formatted_entry_txt,
                entry_md=formatted_entry_md,
                page_link=source.get("page_link"),
                similarity=round(similarity, 4),
            )
        )
        if len(results) >= max_results:
            break

    results.sort(key=lambda item: item.similarity, reverse=True)
    return DictionaryResponse(results=results)


def process_batch(texts):
    input_texts = texts
    inputs = tokenizer(input_texts,
                       #return_tensors="pt",
                       #padding=True,
                       truncation=True,
                       max_length=max_length)
    # Convert to list of token lists
    input_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']]
    #input_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in input_texts]
    results = translator.translate_batch(input_tokens, max_decoding_length=512, beam_size=1, max_batch_size=200)
    output_texts = [tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0])) for result in results]
    return output_texts

@app.post("/tagging/", response_model=TaggingResponse)
async def tag_texts(request: TaggingRequest):
    results = []
    for i in range(0, len(request.texts), 2000):
        batch = request.texts[i:i+2000]
        results.extend(process_batch(batch))
    return TaggingResponse(results=results)

@app.post("/tagging-parsed/", response_model=TaggerParsedResponse)
async def tag_texts_parsed(request: TaggerParsedRequest):
    """
    Tag texts and return parsed output with human-readable tags.
    
    This endpoint processes Sanskrit texts and returns structured grammatical analysis
    with optional human-readable grammatical tags and dictionary meanings.
    """
    # First, get raw tagging results
    prefix = PREFIX_MAP.get(request.mode.value, "")
    raw_results = []
    for i in range(0, len(request.texts), 2000):
        batch = request.texts[i:i+2000]
        prefixed_texts = [prefix + text for text in batch]
        raw_results.extend(process_batch(prefixed_texts))
    
    # Then postprocess each result
    parsed_results = []
    for text, raw_result in zip(request.texts, raw_results):
        # Strip the prefix that was added for processing
        # The text should have the original sentence without prefix
        grammatical_analysis = postprocessor.postprocess_sanskrit(
            raw_result, 
            mode=request.mode.value,
            human_readable_tags=request.human_readable_tags,
            grammar_type=request.grammar_type.value
        )
        
        # For segmentation-lemma-morphosyntax mode (unsandhied-lemma-morphosyntax in endpoint),
        # only return grammar analysis if more than 50% of lemmas have dictionary entries
        if request.mode.value == "unsandhied-lemma-morphosyntax":
            if grammatical_analysis:
                lemmas_with_entries = sum(1 for item in grammatical_analysis if item.get("meanings") and len(item.get("meanings", [])) > 0)
                total_lemmas = len(grammatical_analysis)
                if total_lemmas > 0:
                    entry_ratio = lemmas_with_entries / total_lemmas
                    if entry_ratio <= 0.5:
                        # Less than or equal to 50% have entries, return empty analysis
                        grammatical_analysis = []
        
        parsed_results.append(
            Sentence(
                sentence=text,
                grammatical_analysis=grammatical_analysis
            )
        )
    
    return TaggerParsedResponse(root=parsed_results)
    
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("APP_PORT", "3415"))
    uvicorn.run(app, host="0.0.0.0", port=port)
