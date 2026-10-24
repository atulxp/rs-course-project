import numpy as np
import pandas as pd
import pytest

from src.data.parsing import parse_area, parse_price, parse_pydict, parse_id_list, parse_floor_num


def test_parse_price_crore():
    lo, hi = parse_price("3.25 Cr")
    assert lo == hi == 32_500_000


def test_parse_price_lakh_range():
    lo, hi = parse_price("69.3  - 83.7 L")
    assert lo == 6_930_000
    assert hi == 8_370_000


def test_parse_price_on_request():
    lo, hi = parse_price("Price on Request")
    assert lo is None and hi is None


def test_parse_price_onwards():
    lo, hi = parse_price("19.1 L Onwards")
    assert lo == pytest.approx(1_910_000)
    assert hi is None


def test_parse_area_range():
    lo, hi = parse_area("518-623  sq.ft.")
    assert lo == 518
    assert hi == 623


def test_parse_area_single():
    lo, hi = parse_area("1089 sq.ft.")
    assert lo == hi == 1089


def test_parse_pydict_valid():
    d = parse_pydict("{'CITY_NAME': 'Gurgaon', 'LOCALITY_NAME': 'Sector 84 Gurgaon'}")
    assert d["CITY_NAME"] == "Gurgaon"


def test_parse_pydict_missing():
    assert parse_pydict(np.nan) == {}
    assert parse_pydict(None) == {}
    assert parse_pydict("not a dict") == {}


def test_parse_id_list():
    assert parse_id_list("32,44,23") == ["32", "44", "23"]
    assert parse_id_list(np.nan) == []
    assert parse_id_list("") == []


def test_parse_floor_num():
    assert parse_floor_num("14") == 14
    assert parse_floor_num("G") == 0
    assert parse_floor_num("B") == -1
    assert parse_floor_num(np.nan) is None


def test_properties_schema(properties):
    required = {"property_id", "city", "transaction_type", "price", "area_sqft", "bhk", "locality"}
    assert required.issubset(set(properties.columns))


def test_properties_no_duplicate_ids(properties):
    assert properties["property_id"].duplicated().sum() == 0


def test_properties_transaction_type_values(properties):
    assert set(properties["transaction_type"].unique()) <= {"Buy", "Rent", "Unknown"}


def test_properties_price_positive_where_valid(properties):
    valid = properties[properties["is_price_valid"]]
    assert (valid["price"] > 0).all()


def test_properties_multiple_cities_present(properties):
    assert properties["city"].nunique() >= 2
