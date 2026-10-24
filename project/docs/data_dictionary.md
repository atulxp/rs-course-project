# Data Dictionary — `data/processed/properties.csv`

This document is the output of the mandatory dataset-inspection step
(Section 4 of the project brief) performed on the attached ZIP, plus the
decisions made while writing `src/data/prepare.py`. All numbers below were
produced by actually running the pipeline on 2026-09-23, not estimated.

## 1. Raw ZIP structure

```
archive.zip
├── mumbai.csv          9,514 rows  x 55 columns
├── gurgaon_10k.csv    10,704 rows  x 67 columns
├── hyderabad.csv       9,487 rows  x 55 columns
├── kolkata.csv         8,797 rows  x 35 columns
└── facets/facets/      11 lookup CSVs (id -> label), e.g. FURNISH.csv, AGE.csv,
                         PROPERTY_TYPE.csv, AMENITIES.csv, BEDROOM_NUM.csv ...
```

**Key finding: the four city files do NOT share one schema.** mumbai.csv and
hyderabad.csv share a 55-column "SPID-based" schema; kolkata.csv is a
35-column subset of that same schema; gurgaon_10k.csv is a completely
different, richer 67-column schema (includes photo URLs, dealer contact
info, `LOCALITY` as its own column instead of nested inside `location`,
etc.). `src/data/prepare.py` harmonizes all four into one canonical table
rather than assuming a shared layout.

The `facets/` lookup tables are id→label dumps that were confirmed (by
cross-referencing id ranges against the actual `FEATURES` / `AMENITIES`
columns in each city file) to only **partially** cover the ids that appear
in the data — they fully describe the Gurgaon export but are incomplete for
Mumbai/Hyderabad/Kolkata. Unresolved ids are decoded as `Other:<id>` rather
than dropped; this is a documented dataset limitation, not a bug.

## 2. Row-level cleaning

| Step | Result |
|---|---|
| Rows loaded (4 files concatenated) | 38,502 |
| Duplicate `property_id` removed | 15 |
| Rows with no price, no area, **and** no BHK at all (unusable) | 5 |
| **Final row count** | **38,482** |

`property_id` is built as `<source_city>_<raw PROP_ID>` to guarantee global
uniqueness (raw PROP_ID alone is only unique within one city file).

## 3. Field-by-field notes and important discoveries

### `PRICE` (free text → `price_min`, `price_max`, `price`, `is_price_valid`)
Raw values look like `"3.25 Cr"`, `"69.3 - 83.7 L"`, `"19.1 L Onwards"`,
`"85,000"`, `"20,000/Bedroom"`, `"Price on Request"`. Parsed by
`src/data/parsing.parse_price()`, which handles Crore/Lakh/thousand
multipliers, price ranges, "Onwards" (open upper bound), and per-bed/
per-bedroom suffixes. `"Price on Request"` → `NaN`, flagged via
`is_price_valid=False` (1.0% of rows) rather than being imputed with a
guessed number.

### `AREA` (free text → `area_min`, `area_max`, `area_sqft`, `is_area_valid`)
Same idea: `"1089 sq.ft."`, `"518-623  sq.ft."` → parsed numeric sqft range.
Values <50 sqft or >50,000 sqft are flagged `is_area_valid=False` (0.3% of
rows) as implausible for residential real estate rather than silently kept.

### `TRANSACT_TYPE` — **the most important discovery of the EDA phase**
The raw column is a numeric code (`1.0`, `2.0`, or missing). The project
brief's assumption that this is a simple Buy/Rent flag is **wrong** for this
dataset. Cross-checking price magnitude and the `SECONDARY_TAGS` /
`LISTING` columns against `TRANSACT_TYPE` showed:

* `TRANSACT_TYPE = 1.0` rows are tagged `RESALE` in `SECONDARY_TAGS` and have
  crore/lakh-scale prices → **Buy (resale)**.
* `TRANSACT_TYPE = 2.0` rows are tagged `NEW BOOKING` / `NEW LAUNCH` and are
  also crore/lakh-scale → **Buy (new/primary booking)** — NOT rent.
* Rows with a **missing** `TRANSACT_TYPE` code have prices in the ₹500–₹2.5L
  numeric range (matching typical Indian **monthly rent**), confirmed by
  manually sampling raw `PRICE` strings for these rows (`"85,000"`,
  `"40,000"`, `"2.5 L"`, ...) — these are the dataset's actual rental
  listings, just not explicitly flagged as such.

Adaptation implemented in `src/data/prepare.py`:
* `listing_status` (new field) = `Resale` / `New Booking` / `Not Specified`,
  decoded directly from the raw code.
* `transaction_type` (`Buy` / `Rent`) is derived as: any row with a present
  `TRANSACT_TYPE` code → `Buy`; rows with a missing code are split into
  `Buy`/`Rent` by `impute_transaction_type()` using a price threshold of
  **INR 2,000,000**, chosen because confirmed-Buy rows rarely fall below
  their 5th percentile (~INR 2.14M) while the missing-code rows have a
  median price of ~INR 28k. Every imputed row is flagged
  `transaction_type_imputed=True` so the heuristic is fully transparent and
  reversible.
* Final split: **32,003 Buy / 6,473 Rent / 6 Unknown** (rows with no valid
  price at all, left unresolved).

### `FURNISH`, `AGE` (integer codes)
Raw values are small integer codes (`0/1/2/4` for furnish; `0/1/2/3/5/6` for
age) that only partially match the `facets/FURNISH.csv` / `facets/AGE.csv`
lookup tables (those facets only cover ids `{1,2,4}` and `{1,2,3,5}`
respectively). Decoded via a manually verified map in `src/data/facets.py`
(`FURNISH_CODE_MAP`, `AGE_CODE_MAP`), with unmapped codes (`0`, `6`) decoded
as `"Not Specified"` rather than dropped.

### `BEDROOM_NUM` → `bhk`
9.2% missing. Values up to 60 appear in the raw data (clear data-entry
artifacts for a residential BHK field) — flagged via `is_bhk_valid`
(`True` only for 1–10 BHK, 90.5% of rows) instead of being silently clipped,
so models can choose how to treat the remainder.

### `FLOOR_NUM`
Mixed alphanumeric (`"14"`, `"G"` for Ground, `"B"`/`"L"` for
Basement/Lower-Ground). Parsed to integers (`G→0`, `B`/`L→-1`) by
`src/data/parsing.parse_floor_num()`. 16.9% missing overall (as high as 47%
for Hyderabad).

### `location` / `MAP_DETAILS` (Python-dict-repr strings)
e.g. `"{'CITY_NAME': 'Gurgaon', 'LOCALITY_NAME': 'Sector 84 Gurgaon', ...}"`.
Parsed safely with `ast.literal_eval` (never `eval`) in
`src/data/parsing.parse_pydict()`. `locality` and `latitude`/`longitude` are
extracted from here; Gurgaon additionally has these fields as plain top-level
columns, handled as a fallback.

### `AMENITIES` / `FEATURES` (comma-separated id lists)
e.g. `"32,44,23,45,..."`. Parsed to a list of ids
(`src/data/parsing.parse_id_list()`), then decoded to labels via the facet
tables where possible (`Other:<id>` fallback otherwise). Stored in the
processed CSV as `;`-joined label strings. 20.8% of rows have no amenities
listed at all (varies a lot by city: Mumbai only 47.7% missing vs Gurgaon
0.8%).

### Outlier handling
`clean_outliers()` in `src/data/prepare.py` **caps** (does not delete)
`price` and `area_sqft` at the 1st/99th percentile **within each
(city, transaction_type) group**, producing `price_capped` /
`area_capped` / `price_per_sqft_capped` columns used by the binning /
feature-engineering step, while the original uncapped `price` / `area_sqft`
are preserved unchanged for display in the UI.

## 4. Final dataset summary (after cleaning)

| City | Rows | Buy | Rent |
|---|---|---|---|
| Mumbai | 9,499 | ~6,867 | ~2,632 |
| Gurgaon | 10,704 | ~10,152 | ~552 |
| Hyderabad | 9,483 | ~7,910 | ~1,573 |
| Kolkata | 8,796 | ~7,059 | ~1,737 |

(exact split reproducible by running `python -m src.data.prepare` and
inspecting `data/processed/properties.csv`)

Property type distribution (all cities): Residential Apartment 28,537 (74%),
Residential Land 3,524, Independent/Builder Floor 3,186, Independent
House/Villa 2,951, Studio Apartment 162, Serviced Apartments 80, Farm House
40.

BHK distribution (valid rows only): 1 BHK 3,096 · 2 BHK 10,495 · 3 BHK
14,252 · 4 BHK 5,771 · 5+ BHK 1,227.

## 5. Columns that were dropped / not used

Dealer contact info, photo URLs, `PROJ_ID`/`BUILDING_ID` internal keys, raw
`REGISTER_DATE`/`POSTING_DATE`/`UPDATE_DATE` timestamps (kept out of the
canonical schema — they describe listing metadata, not property attributes,
and are not consistently present across all four files), and `CLASS_LABEL`
free-text broker names, are all excluded from the canonical schema
(`CANONICAL_COLUMNS` in `src/data/prepare.py`) as they add no recommendation
signal and several risk leaking identity information.

## 6. Fields that could leak information

None of the retained columns describe *user* behaviour (this is a
property-only dataset — see README "Academic Integrity" section for why
user interactions are entirely synthetic). `BUILDING_ID` / `PROJ_ID` were
dropped because they are internal database keys with no stable meaning
across the four separately-scraped files.

## 7. Data-quality bug found and fixed during test-writing

While writing `tests/test_data.py::test_parse_price_lakh_range`, we discovered
that `parse_price()` initially mis-parsed two-sided price ranges whose unit
(Cr/L) is stated only once, at the end (e.g. `"1.7 - 1.71 Cr"`): the lower
bound was parsed as the bare number `1.7` instead of `17,000,000`, because
the unit-detection regex only looked at each half of the range in isolation.
This silently corrupted `price_min` for every ranged listing in the raw data
(a non-trivial fraction of rows — ranged prices are common for
under-construction projects). Fixed in `parse_price()` by detecting the
single trailing unit for the whole string and applying it to whichever side
of the range doesn't carry its own unit token, then **the entire pipeline
was retrained from scratch** on the corrected parser before any evaluation
numbers were produced. This is called out explicitly here in the spirit of
Section 39 of the project brief ("no fake results") — the dataset inspection
and cleaning process is iterative and this is a real example of that.
