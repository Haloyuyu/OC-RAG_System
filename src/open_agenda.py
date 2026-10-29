"""Chargement et préparation des événements OpenAgenda pour l'indexation FAISS."""

import json
import re

import pandas as pd
import requests
from langchain_core.documents import Document

BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda"

COLUMNS = [
    # identité / texte
    "uid", "title_fr", "description_fr", "longdescription_fr", "keywords_fr",
    "conditions_fr", "originagenda_title", "canonicalurl",
    # dates
    "daterange_fr", "firstdate_begin", "lastdate_end", "updatedat",
    # lieu
    "location_name", "location_address", "location_postalcode", "location_city",
    "location_department", "location_coordinates", "location_website", "location_phone",
    # infos pratiques
    "age_min", "age_max", "status", "attendancemode", "registration",
    "accessibility_label_fr",
]

EXCLUDED_STATUSES = ["Annulé", "Reporté"]

METADATA_KEYS = ["uid", "title", "city", "department", "start", "end", "url"]


# --- 1. Chargement depuis l'API Opendatasoft (miroir OpenAgenda, sans clé) ---

def fetch_events(region="Occitanie", start="2025-01-01", end="2026-10-29", limit=None):
    """Télécharge les événements de `region` qui chevauchent la fenêtre [start, end]."""
    params = {
        "refine": f"location_region:{region}",
        "where": f"lastdate_end >= date'{start}' AND firstdate_begin <= date'{end}'",
        "select": ",".join(COLUMNS),
    }
    if limit is not None:
        params["limit"] = limit

    resp = requests.get(f"{BASE_URL}/exports/json", params=params, timeout=300)
    resp.raise_for_status()
    return pd.DataFrame(resp.json(), columns=COLUMNS)


# --- 2. Décodage des colonnes JSON texte ---

def is_empty(value):
    """True pour None, NaN, pd.NA, chaîne vide ou liste vide."""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return value is None or pd.isna(value) or value == ""

def label_fr(value):
    """Extrait le libellé français d'un champ JSON {id, label: str | {fr: ...}}."""
    if is_empty(value):
        return None
    label = json.loads(value).get("label")
    if isinstance(label, dict):
        return label.get("fr")
    return label

def registration_links(value):
    """Retourne la liste des liens/contacts d'inscription."""
    if is_empty(value):
        return []
    return [str(r["value"]) for r in json.loads(value) if r.get("value")]

def decode_columns(df):
    """Convertit les colonnes JSON texte en valeurs Python lisibles."""
    df = df.copy()
    df["status"] = df["status"].map(label_fr)                  # "Programmé", "Annulé", ...
    df["attendancemode"] = df["attendancemode"].map(label_fr)  # "Sur place", "En ligne", "Mixte"
    df["registration"] = df["registration"].map(registration_links)
    return df


# --- 3. Nettoyage ---

def clean_events(df):
    """Retire les événements annulés/reportés, sans titre français et les doublons."""
    df = df[~df["status"].isin(EXCLUDED_STATUSES)]
    df = df[~df["title_fr"].map(is_empty)]  # quelques événements saisis uniquement en occitan
    return df.drop_duplicates(subset="uid")

def clean_html(text):
    """Retire les balises HTML et normalise les espaces."""
    if is_empty(text):
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


# --- 4. Construction des documents LangChain ---

def build_page_content(row):
    """Texte vectorisé : tout ce sur quoi l'utilisateur peut poser une question."""
    lieu = ", ".join(
        str(row[c]) for c in ["location_name", "location_address", "location_city", "location_department"]
        if not is_empty(row[c])
    )
    fields = {
        "Titre": row["title_fr"],
        "Dates": row["daterange_fr"],
        "Lieu": lieu,
        "Organisateur": row["originagenda_title"],
        "Résumé": clean_html(row["description_fr"]),
        "Description": clean_html(row["longdescription_fr"]),
        "Mots-clés": ", ".join(row["keywords_fr"]) if isinstance(row["keywords_fr"], list) else None,
        "Conditions": row["conditions_fr"],
        "Âge": f"de {int(row['age_min'])} à {int(row['age_max'])} ans"
               if not is_empty(row["age_min"]) and not is_empty(row["age_max"]) else None,
        "Mode": row["attendancemode"],
        "Statut": row["status"],
        "Accessibilité": row["accessibility_label_fr"],
        "Réservation": ", ".join(row["registration"]) or None,
        "Lien": row["canonicalurl"],
    }
    return "\n".join(f"{k} : {v}" for k, v in fields.items() if not is_empty(v))

def none_if_empty(value):
    return None if is_empty(value) else value

def build_documents(df):
    """Un Document LangChain par événement, prêt pour FAISS.from_documents."""
    return [
        Document(
            page_content=build_page_content(row),
            metadata={
                "uid": row["uid"],
                "title": none_if_empty(row["title_fr"]),
                "city": none_if_empty(row["location_city"]),
                "department": none_if_empty(row["location_department"]),
                "start": none_if_empty(row["firstdate_begin"]),
                "end": none_if_empty(row["lastdate_end"]),
                "url": none_if_empty(row["canonicalurl"]),
            },
        )
        for _, row in df.iterrows()
    ]

def prepare_documents(raw_df):
    """Pipeline complet : DataFrame brut de l'API -> (DataFrame propre, documents)."""
    df = clean_events(decode_columns(raw_df))
    return df, build_documents(df)
