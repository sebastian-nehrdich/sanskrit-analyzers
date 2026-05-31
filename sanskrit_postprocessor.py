import os
from difflib import SequenceMatcher
import pandas as pd
from typing import List, Dict, Optional

from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions
from endpoint_models import Meaning


def read_skt_tags(path: str) -> Dict[str, str]:
    """Read Sanskrit tags from a TSV file."""
    result = {}
    lines = open(path).readlines()
    lines = [line.split("\t") for line in lines]
    for line in lines:
        if len(line) >= 2:
            result[line[0]] = line[1].strip()
    return result


def read_skt_dict(path: str) -> Dict[str, List[str]]:
    """Read Sanskrit dictionary from a CSV file."""
    df = pd.read_csv(path, sep="\t")
    # create dictionary with 'word' as key and 'meanings' as value
    # when there are multiple values, append them as a list
    skt_dict = {}
    for index, row in df.iterrows():
        if row['word'] in skt_dict:
            if type(skt_dict[row['word']]) == list:
                skt_dict[row['word']].append(row['meanings'])
            else:
                skt_dict[row['word']] = [skt_dict[row['word']], row['meanings']]
        else:
            skt_dict[row['word']] = [row['meanings']]
    
    return skt_dict


def human_readable_tag(tag: str) -> str:
    """Convert abbreviated grammatical tags to human-readable format."""
    tag = tag.replace("|", ", ")
    tag = tag.replace("Case=Cpd", "Compound")
    tag = tag.replace("Mood=Ind", "Mood=Indicative")
    tag = tag.replace("Ins", "Instrumental")
    tag = tag.replace("Loc", "Locative")
    tag = tag.replace("Abl", "Ablative")
    tag = tag.replace("Dat", "Dative")
    tag = tag.replace("=Gen", "=Genitive")
    tag = tag.replace("Acc", "Accusative")
    tag = tag.replace("Nom", "Nominative")
    tag = tag.replace("Voc", "Vocative")
    tag = tag.replace("Prs", "Present")
    tag = tag.replace("Fut", "Future")
    tag = tag.replace("Pst", "Past")
    tag = tag.replace("Imp", "Imperative")
    tag = tag.replace("Opt", "Optative")
    tag = tag.replace("Cond", "Conditional")
    tag = tag.replace("Pot", "Potential")
    tag = tag.replace("Perf", "Perfect")
    tag = tag.replace("Pres", "Present")
    tag = tag.replace("Masc", "Masculine")
    tag = tag.replace("Fem", "Feminine")
    tag = tag.replace("Neut", "Neuter")
    tag = tag.replace("Sing", "Singular")
    tag = tag.replace("Plur", "Plural")
    tag = tag.replace("Pass", "Passive")
    tag = tag.replace("Act", "Active")
    tag = tag.replace("Mid", "Middle")
    return tag


def indic_paninian_tag(tag: str) -> str:
    """
    Map Western-style Sanskrit grammatical labels to more Indic/Pāṇinian terms.
    This assumes `tag` is a short label like "Nominative", "Present", "Person=1", etc.
    If your input strings can contain multiple labels, call this repeatedly.
    """
    # First apply human_readable_tag to get full forms
    tag = human_readable_tag(tag)
    
    # --- cases (vibhakti) ---
    replacements = [
        ("Nominative", "प्रथामा-विभक्ति"),   # or "प्रथमा"
        ("Accusative", "द्वितीया-विभक्ति"),
        ("Instrumental", "तृतीया-विभक्ति"),
        ("Dative", "चतुर्थी-विभक्ति"),
        ("Ablative", "पञ्चमी-विभक्ति"),
        ("Genitive", "षष्ठी-विभक्ति"),
        ("Locative", "सप्तमी-विभक्ति"),
        ("Vocative", "सम्बोधन"),  # usually not counted as a numbered vibhakti
    ]

    # --- number (vacana) ---
    replacements += [
        ("Singular", "एकवचन"),
        ("Dual", "द्विवचन"),
        ("Plural", "बहुवचन"),
    ]

    # --- gender (liṅga) ---
    replacements += [
        ("Masculine", "पुंलिङ्ग"),
        ("Feminine", "स्त्रीलिङ्ग"),
        ("Neuter", "नपुंसकलिङ्ग"),
    ]

    # --- finite verbs: lakāra / tense / mood ---
    # try to map the more specific ones first
    replacements += [
        # specific pasts
        ("Imperfect", "अनद्यतन-भूत (लङ्)"),
        ("Aorist", "लुङ्"),
        ("Perfect", "परोक्ष (लिट्)"),
        ("Periphrastic Future", "लुट्"),
        # more common western labels
        ("Present", "वर्तमान (लट्)"),
        ("Past", "भूतकाल (लङ्)"),           # fallback
        ("Future", "भविष्यत्काल (लृट्)"),
        ("Imperative", "आज्ञार्थक (लोट्)"),
        ("Optative", "विधिलिङ् (लिङ्)"),
        ("Potential", "विधिलिङ् (लिङ्)"),    # western "potential" = लिङ्
        ("Benedictive", "आशीर्लिङ्"),
        ("Conditional", "शर्ते लृङ् (लृङ्)"),
    ]

    # --- voice / pada / prayoga ---
    replacements += [
        ("Active", "कर्तरि-प्रयोग"),
        ("Middle", "आत्मनेपद"),
        ("Passive", "कर्मणि-प्रयोग"),
        # if source ever uses these directly
        ("Parasmaipada", "परस्मैपद"),
        ("Atmanepada", "आत्मनेपद"),
    ]

    # --- "mood" / misc western labels ---
    # "Indicative" is fuzzy; keep it very neutral
    replacements += [
        ("Indicative", "साधारण रूप"),
    ]

    # --- non-finite forms ---
    replacements += [
        ("Compound", "समास"),
        ("Participle", "कृदन्त"),
        ("Gerundive", "कृत्य"),
        ("Infinitive", "तुमुन्-न्त"),
        ("Absolutive", "क्त्वा / ल्यप्"),
    ]

    # --- person (puruṣa) ---
    # NB: Indian order is 3 → प्रथम, 2 → मध्यम, 1 → उत्तम
    replacements += [
        ("Person=1", "उत्तम-पुरुष"),
        ("Person=2", "मध्यम-पुरुष"),
        ("Person=3", "प्रथम-पुरुष"),
    ]

    # apply all
    for src, tgt in replacements:
        if src in tag:
            tag = tag.replace(src, tgt)

    return tag


class SanskritPostprocessor:
    """Handles postprocessing of Sanskrit grammar analysis."""
    
    def __init__(self, tags_path: str = "data/sanskrit_tags.tsv", 
                 dict_path: str = "data/dictionary.csv",
                 es_url: Optional[str] = None,
                 es_index: Optional[str] = None):
        """Initialize the postprocessor with tag and dictionary data."""
        self.sanskrit_tags = read_skt_tags(tags_path)
        self.skt_dict = read_skt_dict(dict_path)
        self._es_url = es_url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self._es_index = es_index or os.getenv("ES_DICTIONARY_INDEX", "sanskrit_dictionaries")
        self._es_client: Optional[Elasticsearch] = None
        self._es_cache: Dict[str, List[Meaning]] = {}
        
        # Mapping of source names to their full dictionary URLs
        self._full_dict_urls = {
            "ap90": "https://www.sanskrit-lexicon.uni-koeln.de/scans/AEScan/2020/web/index.php",
            "mw72": "https://www.sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/web/index.php",
            "pwg": "https://www.sanskrit-lexicon.uni-koeln.de/scans/PWScan/2020/web/index.php",
            "bhs": "https://www.sanskrit-lexicon.uni-koeln.de/scans/BHSScan/2020/web/index.php",
        }

    def _get_es_client(self) -> Optional[Elasticsearch]:
        if not self._es_index:
            return None
        if self._es_client is None:
            try:
                self._es_client = Elasticsearch(self._es_url)
            except Exception:
                self._es_client = None
        return self._es_client

    def _get_full_dict_url(self, source: str | None) -> str | None:
        """Get the full dictionary URL for a source, if available."""
        if not source:
            return None
        source_lower = source.lower()
        # Check for exact match first
        if source_lower in self._full_dict_urls:
            return self._full_dict_urls[source_lower]
        # Check if source starts with any key (for cases like "ap90-something")
        for key, url in self._full_dict_urls.items():
            if source_lower.startswith(key):
                return url
        return None

    def _format_meaning(self, meaning_text: str, source: str | None = None, page_link: str | None = None) -> str:
        """Format meaning text with source heading, page link, and full dictionary link."""
        formatted_parts = []
        if source:
            formatted_parts.append(f"## {source}")
        
        # Add links (PDF page and Full Dictionary)
        link_parts = []
        if page_link:
            link_parts.append(f"[Open PDF page]({page_link})")
        full_dict_url = self._get_full_dict_url(source)
        if full_dict_url:
            link_parts.append(f"[Full Dictionary]({full_dict_url})")
        
        if link_parts:
            formatted_parts.append(" | ".join(link_parts))
        
        if formatted_parts:
            formatted_parts.append("")  # Add blank line before content
        formatted_parts.append(meaning_text)
        return "\n".join(formatted_parts)

    def _collapse_meanings_by_source(self, meanings: List[Meaning]) -> List[Meaning]:
        """Collapse multiple entries from the same dictionary into one entry with visual separation."""
        from collections import defaultdict
        
        # Define source priority order: ap90, mw72, pwg, bhs, then others
        SOURCE_PRIORITY = {
            "ap90": 0,
            "mw72": 1,
            "pwg": 2,
            "bhs": 3,
        }
        
        def get_source_priority(source: str) -> int:
            """Get priority for a source. Lower number = higher priority."""
            if not source:
                return 999
            source_lower = source.lower()
            # Check for exact match first (case-insensitive)
            if source_lower in SOURCE_PRIORITY:
                return SOURCE_PRIORITY[source_lower]
            # Check if source starts with any priority key (for cases like "ap90-something")
            for key, priority in SOURCE_PRIORITY.items():
                if source_lower.startswith(key):
                    return priority
            # Sources not in priority list come after
            return 999
        
        # Group meanings by source
        meanings_by_source = defaultdict(list)
        for meaning in meanings:
            source = meaning.source or "Unknown"
            meanings_by_source[source].append(meaning)
        
        # Sort sources by priority
        sorted_sources = sorted(meanings_by_source.keys(), key=get_source_priority)
        
        collapsed_meanings = []
        for source in sorted_sources:
            source_meanings = meanings_by_source[source]
            if len(source_meanings) == 1:
                # Single entry: keep as is
                collapsed_meanings.append(source_meanings[0])
            else:
                # Multiple entries: collapse into one
                subentries = []
                for meaning in source_meanings:
                    # Extract the actual content (remove the source heading and page link if present)
                    # The formatted meaning has structure: "## {source}\n[Open PDF page](...)\n\n{content}"
                    # or "## {source}\n\n{content}" if no page link
                    lines = meaning.meaning.split("\n")
                    content_lines = []
                    skip_until_content = True
                    
                    for line in lines:
                        line_stripped = line.strip()
                        # Skip header line
                        if line_stripped == f"## {source}":
                            continue
                        # Skip link lines (PDF page and Full Dictionary)
                        if line_stripped.startswith("[Open PDF page]") or line_stripped.startswith("[Full Dictionary]"):
                            continue
                        # Skip lines that contain both links (separated by |)
                        if "|" in line_stripped and ("[Open PDF page]" in line_stripped or "[Full Dictionary]" in line_stripped):
                            continue
                        # Skip blank lines that come before content
                        if skip_until_content and not line_stripped:
                            continue
                        # Once we hit non-blank content, start collecting
                        if line_stripped:
                            skip_until_content = False
                        if not skip_until_content:
                            content_lines.append(line)
                    
                    # Get the actual meaning text
                    content = "\n".join(content_lines).strip()
                    
                    # Format subentry with page link and full dictionary link at the beginning
                    subentry_parts = []
                    link_parts = []
                    if meaning.page_link:
                        link_parts.append(f"[Open PDF page]({meaning.page_link})")
                    full_dict_url = self._get_full_dict_url(source)
                    if full_dict_url:
                        link_parts.append(f"[Full Dictionary]({full_dict_url})")
                    
                    if link_parts:
                        subentry_parts.append(f"**Page:** {' | '.join(link_parts)}")
                    subentry_parts.append(content)
                    subentries.append("\n".join(subentry_parts))
                
                # Combine subentries with visual separation
                header_parts = [f"## {source}"]
                header_link_parts = []
                if source_meanings[0].page_link:
                    header_link_parts.append(f"[Open PDF page]({source_meanings[0].page_link})")
                full_dict_url = self._get_full_dict_url(source)
                if full_dict_url:
                    header_link_parts.append(f"[Full Dictionary]({full_dict_url})")
                
                if header_link_parts:
                    header_parts.append(" | ".join(header_link_parts))
                
                combined_content = "\n".join(header_parts) + "\n\n" + "\n\n---\n\n".join(subentries)
                collapsed_meanings.append(Meaning(
                    meaning=combined_content,
                    source=source,
                    page_link=source_meanings[0].page_link  # Keep first page link for reference
                ))
        
        return collapsed_meanings

    def _lookup_es_entries(self, lemma: str) -> List[Meaning]:
        if lemma in self._es_cache:
            return self._es_cache[lemma]

        client = self._get_es_client()
        if client is None:
            self._es_cache[lemma] = []
            return []

        try:
            response = client.search(
                index=self._es_index,
                size=20,
                query={
                    "bool": {
                        "should": [
                            {"term": {"key.keyword": lemma}},
                            {"term": {"keyword": lemma}},
                            {
                                "match": {
                                    "key": {
                                        "query": lemma,
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
        except es_exceptions.ElasticsearchException:
            self._es_cache[lemma] = []
            return []

        results: List[Meaning] = []
        lemma_lower = lemma.lower()
        for hit in hits:
            source_data = hit.get("_source", {})
            key = source_data.get("key") or source_data.get("keyword")
            if not key:
                continue
            similarity = SequenceMatcher(None, lemma_lower, key.lower()).ratio()
            if similarity < 0.99:
                continue
            entry_md = source_data.get("entry_md")
            entry_txt = source_data.get("entry_txt")
            source_name = source_data.get("source")
            page_link = source_data.get("page_link")
            
            # Prefer markdown version, fall back to text if markdown not available
            meaning_text = entry_md or entry_txt
            if meaning_text:
                formatted_meaning = self._format_meaning(meaning_text, source_name, page_link)
                results.append(Meaning(
                    meaning=formatted_meaning,
                    source=source_name,
                    page_link=page_link
                ))
        self._es_cache[lemma] = results
        return results
    
    def postprocess_sanskrit(self, sentence: str, mode: str = "lemma-morphosyntax", 
                           human_readable_tags: bool = False,
                           grammar_type: str = "western") -> List[Dict]:
        """
        Postprocess Sanskrit sentence based on the specified mode.
        
        Args:
            sentence: The tagged sentence to process
            mode: Processing mode (lemma-morphosyntax, unsandhied-lemma-morphosyntax, etc.)
            human_readable_tags: Whether to convert tags to human-readable format
            grammar_type: Type of grammar terminology ("western" or "indic")
            
        Returns:
            List of dictionaries containing grammatical analysis
        """
        if mode == "lemma-morphosyntax" or mode == "unsandhied-lemma-morphosyntax":
            result = []
            for item in sentence.split(" "):
                if mode == "unsandhied-lemma-morphosyntax":
                    if len(item.split("_")) == 3:
                        unsandhied, lemma, short_tag = item.split("_")
                        if short_tag in self.sanskrit_tags:
                            short_tag = self.sanskrit_tags[short_tag]
                            if "Cpd" in short_tag:
                                unsandhied = unsandhied + "-"
                            if human_readable_tags:
                                if grammar_type == "indic":
                                    short_tag = indic_paninian_tag(short_tag)
                                else:
                                    short_tag = human_readable_tag(short_tag)
                        es_meanings = self._lookup_es_entries(lemma)
                        dict_meanings = [Meaning(meaning=self._format_meaning(m, "DCS"), source="DCS") for m in (self.skt_dict[lemma] if lemma in self.skt_dict else [])]
                        meanings = self._collapse_meanings_by_source(es_meanings + dict_meanings)
                        result.append({
                            "unsandhied": unsandhied,
                            "lemma": lemma,
                            "tag": short_tag,
                            "meanings": meanings
                        })
                else:
                    if len(item.split("_")) == 2:
                        lemma, short_tag = item.split("_")
                        if short_tag in self.sanskrit_tags:
                            short_tag = self.sanskrit_tags[short_tag]
                            if human_readable_tags:
                                if grammar_type == "indic":
                                    short_tag = indic_paninian_tag(short_tag)
                                else:
                                    short_tag = human_readable_tag(short_tag)
                        es_meanings = self._lookup_es_entries(lemma)
                        dict_meanings = [Meaning(meaning=self._format_meaning(m, "DCS"), source="DCS") for m in (self.skt_dict[lemma] if lemma in self.skt_dict else [])]
                        meanings = self._collapse_meanings_by_source(es_meanings + dict_meanings)
                        result.append({
                            "lemma": lemma,
                            "tag": short_tag,
                            "meanings": meanings
                        })
            return result
            
        elif mode == "lemma":
            result = []
            for item in sentence.split("_"):
                if len(item) > 0:
                    es_meanings = self._lookup_es_entries(item)
                    dict_meanings = [Meaning(meaning=self._format_meaning(m, "DCS"), source="DCS") for m in (self.skt_dict[item] if item in self.skt_dict else [])]
                    meanings = self._collapse_meanings_by_source(es_meanings + dict_meanings)
                    result.append({
                        "unsandhied": "",
                        "lemma": item,
                        "tag": "",
                        "meanings": meanings
                    })
            return result
            
        elif mode == "unsandhied":
            return [{
                "unsandhied": item,
                "lemma": "",
                "tag": "",
                "meanings": []
            } for item in sentence.split("_") if len(item) > 0]
        
        return []

