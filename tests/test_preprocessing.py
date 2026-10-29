"""Tests unitaires des fonctions de décodage, nettoyage et construction des documents."""

import json

import numpy as np
import pandas as pd
import pytest

from open_agenda import (
    build_page_content,
    clean_events,
    clean_html,
    decode_columns,
    is_empty,
    label_fr,
    none_if_empty,
    registration_links,
)

from conftest import raw_event


class TestIsEmpty:
    @pytest.mark.parametrize("value", [None, np.nan, pd.NA, "", [], {}])
    def test_empty_values(self, value):
        assert is_empty(value)

    @pytest.mark.parametrize("value", ["texte", 0, 0.0, ["a"], {"a": 1}, False])
    def test_non_empty_values(self, value):
        assert not is_empty(value)


class TestLabelFr:
    def test_label_dict(self):
        assert label_fr('{"id": 1, "label": {"fr": "Sur place", "en": "In situ"}}') == "Sur place"

    def test_label_plain_string(self):
        assert label_fr('{"id": 2, "label": "En ligne"}') == "En ligne"

    def test_label_without_french(self):
        assert label_fr('{"id": 1, "label": {"en": "Scheduled"}}') is None

    @pytest.mark.parametrize("value", [None, np.nan, ""])
    def test_missing_value(self, value):
        assert label_fr(value) is None


class TestRegistrationLinks:
    def test_extracts_values(self):
        value = json.dumps([{"type": "link", "value": "https://a.fr"}, {"type": "phone", "value": "0102030405"}])
        assert registration_links(value) == ["https://a.fr", "0102030405"]

    def test_skips_null_values(self):
        value = json.dumps([{"type": "link", "value": "https://a.fr"}, {"type": "phone", "value": None}])
        assert registration_links(value) == ["https://a.fr"]

    @pytest.mark.parametrize("value", [None, np.nan, ""])
    def test_missing_value(self, value):
        assert registration_links(value) == []


class TestCleanHtml:
    def test_removes_tags(self):
        assert clean_html("<p>Venez <b>nombreux</b> !</p>") == "Venez nombreux !"

    def test_normalizes_whitespace(self):
        assert clean_html("ligne 1\n\n   ligne\t2") == "ligne 1 ligne 2"

    @pytest.mark.parametrize("value", [None, np.nan, ""])
    def test_missing_value(self, value):
        assert clean_html(value) == ""


def test_none_if_empty():
    assert none_if_empty(np.nan) is None
    assert none_if_empty("Toulouse") == "Toulouse"


def test_decode_columns_does_not_mutate_input(raw_sample):
    original = raw_sample.copy()
    decode_columns(raw_sample)
    pd.testing.assert_frame_equal(raw_sample, original)


def test_clean_events_removes_excluded_untitled_and_duplicates(raw_sample):
    df = clean_events(decode_columns(raw_sample))
    assert sorted(df["uid"]) == ["1", "2", "3"]


class TestBuildPageContent:
    def row(self, **overrides):
        return decode_columns(pd.DataFrame([raw_event(**overrides)])).iloc[0]

    def test_contains_key_fields(self):
        content = build_page_content(self.row())
        assert content.startswith("Titre : Concert de jazz")
        assert "Lieu : Jardin des Plantes, 31 allée Jules Guesde, Toulouse, Haute-Garonne" in content
        assert "Mots-clés : jazz, concert" in content
        assert "Âge : de 0 à 99 ans" in content
        assert "Mode : Sur place" in content
        assert "Réservation : https://billetterie.fr" in content
        assert "Lien : https://openagenda.com/e/1" in content

    def test_html_is_cleaned(self):
        assert "Description : Venez nombreux !" in build_page_content(self.row())

    def test_missing_fields_are_omitted(self):
        content = build_page_content(self.row(
            conditions_fr=None, keywords_fr=None, age_min=np.nan, age_max=np.nan, registration=None,
        ))
        for label in ["Conditions", "Mots-clés", "Âge", "Réservation"]:
            assert f"{label} :" not in content

    def test_partial_location(self):
        content = build_page_content(self.row(location_name=None, location_address=None))
        assert "Lieu : Toulouse, Haute-Garonne" in content
