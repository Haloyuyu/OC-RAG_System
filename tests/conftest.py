import json

import numpy as np
import pandas as pd
import pytest

from open_agenda import COLUMNS, fetch_events, prepare_documents


def label(fr):
    return json.dumps({"id": 1, "label": {"fr": fr, "en": "..."}})


def raw_event(**overrides):
    """Un enregistrement brut tel que renvoyé par /exports/json."""
    event = {
        "uid": "1",
        "title_fr": "Concert de jazz",
        "description_fr": "Un concert en plein air",
        "longdescription_fr": "<p>Venez <b>nombreux</b> !</p>",
        "keywords_fr": ["jazz", "concert"],
        "conditions_fr": "Gratuit",
        "originagenda_title": "Ville de Toulouse",
        "canonicalurl": "https://openagenda.com/e/1",
        "daterange_fr": "Samedi 14 mars, 20h00",
        "firstdate_begin": "2026-03-14T19:00:00+00:00",
        "lastdate_end": "2026-03-14T22:00:00+00:00",
        "updatedat": "2026-02-01T10:00:00+00:00",
        "location_name": "Jardin des Plantes",
        "location_address": "31 allée Jules Guesde",
        "location_postalcode": "31000",
        "location_city": "Toulouse",
        "location_department": "Haute-Garonne",
        "location_coordinates": {"lon": 1.45, "lat": 43.59},
        "location_website": None,
        "location_phone": None,
        "age_min": 0,
        "age_max": 99,
        "status": label("Programmé"),
        "attendancemode": label("Sur place"),
        "registration": json.dumps([{"type": "link", "value": "https://billetterie.fr"}]),
        "accessibility_label_fr": None,
    }
    event.update(overrides)
    return event


@pytest.fixture
def raw_sample():
    """Échantillon couvrant les cas limites rencontrés dans les vraies données."""
    return pd.DataFrame(
        [
            raw_event(),
            # label en texte simple au lieu d'un dict de traductions
            raw_event(uid="2", attendancemode=json.dumps({"id": 2, "label": "En ligne"}),
                      location_name=None, location_address=None, location_city=None),
            # valeurs manquantes partout + inscription avec value null
            raw_event(uid="3", longdescription_fr=None, keywords_fr=None, conditions_fr=np.nan,
                      age_min=np.nan, age_max=np.nan, attendancemode=None,
                      registration=json.dumps([{"type": "link", "value": "https://a.fr"},
                                               {"type": "phone", "value": None}])),
            # événements à exclure
            raw_event(uid="4", status=label("Annulé")),
            raw_event(uid="5", status=label("Reporté")),
            raw_event(uid="6", title_fr=None, description_fr=None),
            # doublon
            raw_event(uid="1"),
        ],
        columns=COLUMNS,
    )


@pytest.fixture(
    params=[
        "sample",
        pytest.param("api", marks=pytest.mark.integration),
    ]
)
def dataset(request):
    """(df, documents) construits depuis l'échantillon local ou depuis l'API réelle."""
    if request.param == "sample":
        raw = request.getfixturevalue("raw_sample")
    else:
        raw = fetch_events(limit=500)
    return prepare_documents(raw)
