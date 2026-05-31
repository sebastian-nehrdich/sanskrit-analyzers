from pydantic import BaseModel, RootModel, Field
from typing import List
from enum import Enum


class TaggingRequest(BaseModel):
    texts: List[str]


class TaggingResponse(BaseModel):
    results: List[str]


class GrammarModes(str, Enum):
    LEMMA = "lemma"
    LEMMA_MORPHOSYNTAX = "lemma-morphosyntax"
    UNSANDHIED_LEMMA_MORPHOSYNTAX = "unsandhied-lemma-morphosyntax"
    UNSANDHIED = "unsandhied"
    UNSANDHIED_MORPHOSYNTAX = "unsandhied-morphosyntax"


class GrammarType(str, Enum):
    WESTERN = "western"
    INDIC = "indic"


class Meaning(BaseModel):
    meaning: str
    source: str | None = None
    page_link: str | None = None


class Lemma(BaseModel):
    lemma: str = ""
    unsandhied: str = ""
    tag: str = ""
    meanings: List[Meaning] = Field(default_factory=list)


class Sentence(BaseModel):
    sentence: str
    grammatical_analysis: List[Lemma]


class TaggerParsedRequest(BaseModel):
    texts: List[str]
    mode: GrammarModes = GrammarModes.UNSANDHIED_LEMMA_MORPHOSYNTAX
    human_readable_tags: bool = True
    grammar_type: GrammarType = GrammarType.WESTERN


class TaggerParsedResponse(RootModel[List[Sentence]]):
    pass


class DictionaryEntry(BaseModel):
    lemma: str
    source: str | None = None
    entry_html: str | None = None
    entry_txt: str | None = None
    entry_md: str | None = None
    page_link: str | None = None
    similarity: float


class DictionaryResponse(BaseModel):
    results: List[DictionaryEntry]

