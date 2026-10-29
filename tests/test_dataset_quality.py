"""Le jeu de documents est-il propre, structuré et prêt pour FAISS ?

Chaque test tourne sur l'échantillon local et, avec `pytest -m integration`,
sur un extrait réel de l'API.
"""

import re

import pandas as pd
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from open_agenda import EXCLUDED_STATUSES, METADATA_KEYS

# valeur manquante sérialisée en texte : "Conditions : nan", "Mode : None", ...
MISSING_VALUE = re.compile(r" : (nan|NaN|None|<NA>|NaT|\[\])\s*$", re.MULTILINE)
HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(\s[^>]*)?/?>")


def test_not_empty(dataset):
    df, documents = dataset
    assert len(documents) > 0
    assert len(documents) == len(df)


def test_documents_are_langchain_documents(dataset):
    _, documents = dataset
    assert all(isinstance(doc, Document) for doc in documents)


def test_uids_are_present_and_unique(dataset):
    _, documents = dataset
    uids = [doc.metadata["uid"] for doc in documents]
    assert all(isinstance(uid, str) and uid for uid in uids)
    assert len(uids) == len(set(uids))


def test_no_excluded_status(dataset):
    df, _ = dataset
    assert not df["status"].isin(EXCLUDED_STATUSES).any()


def test_page_content_is_non_empty_text(dataset):
    _, documents = dataset
    for doc in documents:
        assert isinstance(doc.page_content, str)
        assert doc.page_content.strip(), doc.metadata["uid"]


def test_page_content_starts_with_title(dataset):
    _, documents = dataset
    for doc in documents:
        assert doc.page_content.startswith("Titre : "), doc.metadata["uid"]


def test_page_content_has_no_missing_value_artifacts(dataset):
    _, documents = dataset
    for doc in documents:
        assert not MISSING_VALUE.search(doc.page_content), (doc.metadata["uid"], doc.page_content)


def test_page_content_has_no_html(dataset):
    _, documents = dataset
    for doc in documents:
        match = HTML_TAG.search(doc.page_content)
        assert match is None, (doc.metadata["uid"], match and match.group())


def test_metadata_schema(dataset):
    _, documents = dataset
    for doc in documents:
        assert sorted(doc.metadata) == sorted(METADATA_KEYS)
        # types simples uniquement : sérialisables et filtrables dans FAISS
        for key, value in doc.metadata.items():
            assert value is None or isinstance(value, str), (doc.metadata["uid"], key, value)


def test_title_and_url_are_always_present(dataset):
    _, documents = dataset
    for doc in documents:
        assert doc.metadata["title"], doc.metadata["uid"]
        assert doc.metadata["url"].startswith("https://"), doc.metadata["uid"]


def test_dates_are_valid_and_ordered(dataset):
    _, documents = dataset
    starts = pd.to_datetime([doc.metadata["start"] for doc in documents], utc=True)
    ends = pd.to_datetime([doc.metadata["end"] for doc in documents], utc=True)
    assert not starts.isna().any()
    assert not ends.isna().any()
    assert (starts <= ends).all()


def test_faiss_index_can_be_built_and_queried(dataset):
    _, documents = dataset
    # embeddings factices : on teste la structure, pas la qualité sémantique
    embeddings = DeterministicFakeEmbedding(size=32)
    index = FAISS.from_documents(documents, embeddings)

    assert index.index.ntotal == len(documents)

    target = documents[0]
    result = index.similarity_search(target.page_content, k=1)[0]
    assert result.metadata == target.metadata
    assert result.page_content == target.page_content
