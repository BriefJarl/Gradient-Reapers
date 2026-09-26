from __future__ import annotations

"""
Feature definitions for Amazon ML Challenge 2026.

Design goals:
- DuckDB-native / vectorized SQL
- No Python row-by-row loops
- No pandas
- Safe NULL handling
- Works for both S2 and S3
- Keeps blocking provenance
- Suitable for ~31M candidate pairs
"""

from typing import Mapping


def resolve_column(
    columns: set[str],
    candidates: list[str],
    required: bool = True,
) -> str | None:
    """
    Resolve a column name from a list of acceptable aliases.
    """
    lower_map = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            "Required column not found.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns: {sorted(columns)}"
        )

    return None


def sql_ident(column: str) -> str:
    """
    Quote a SQL identifier safely.
    """
    return '"' + column.replace('"', '""') + '"'


def build_feature_select(
    s1_alias: str,
    target_alias: str,
    candidate_alias: str,
    s1_columns: set[str],
    target_columns: set[str],
) -> str:
    """
    Build the feature SELECT list.

    The normalized datasets are expected to contain:
      entity_id
      name_norm
      name_compact
      address_norm
      address_compact
      country
      name_tokens
      address_tokens

    ASCII columns are optional. If unavailable, normalized strings
    are used as the fallback.
    """

    # ------------------------------------------------------------
    # Resolve normalized columns
    # ------------------------------------------------------------

    s1_id = resolve_column(
        s1_columns,
        ["entity_id"],
    )

    target_id = resolve_column(
        target_columns,
        ["entity_id"],
    )

    s1_name = resolve_column(
        s1_columns,
        ["name_norm"],
    )

    target_name = resolve_column(
        target_columns,
        ["name_norm"],
    )

    s1_name_compact = resolve_column(
        s1_columns,
        ["name_compact"],
    )

    target_name_compact = resolve_column(
        target_columns,
        ["name_compact"],
    )

    s1_address = resolve_column(
        s1_columns,
        ["address_norm"],
    )

    target_address = resolve_column(
        target_columns,
        ["address_norm"],
    )

    s1_address_compact = resolve_column(
        s1_columns,
        ["address_compact"],
    )

    target_address_compact = resolve_column(
        target_columns,
        ["address_compact"],
    )

    s1_country = resolve_column(
        s1_columns,
        ["country", "country_norm"],
    )

    target_country = resolve_column(
        target_columns,
        ["country", "country_norm"],
    )

    s1_name_tokens = resolve_column(
        s1_columns,
        ["name_tokens"],
    )

    target_name_tokens = resolve_column(
        target_columns,
        ["name_tokens"],
    )

    s1_address_tokens = resolve_column(
        s1_columns,
        ["address_tokens"],
    )

    target_address_tokens = resolve_column(
        target_columns,
        ["address_tokens"],
    )

    # ------------------------------------------------------------
    # Optional ASCII representations
    # ------------------------------------------------------------

    s1_name_ascii = resolve_column(
        s1_columns,
        [
            "name_ascii",
            "name_ascii_norm",
            "name_unidecode",
        ],
        required=False,
    )

    target_name_ascii = resolve_column(
        target_columns,
        [
            "name_ascii",
            "name_ascii_norm",
            "name_unidecode",
        ],
        required=False,
    )

    s1_address_ascii = resolve_column(
        s1_columns,
        [
            "address_ascii",
            "address_ascii_norm",
            "address_unidecode",
        ],
        required=False,
    )

    target_address_ascii = resolve_column(
        target_columns,
        [
            "address_ascii",
            "address_ascii_norm",
            "address_unidecode",
        ],
        required=False,
    )

    # ------------------------------------------------------------
    # SQL expressions
    # ------------------------------------------------------------

    name1 = f"COALESCE({s1_alias}.{sql_ident(s1_name)}, '')"
    name2 = f"COALESCE({target_alias}.{sql_ident(target_name)}, '')"

    namec1 = (
        f"COALESCE({s1_alias}.{sql_ident(s1_name_compact)}, '')"
    )
    namec2 = (
        f"COALESCE({target_alias}.{sql_ident(target_name_compact)}, '')"
    )

    addr1 = (
        f"COALESCE({s1_alias}.{sql_ident(s1_address)}, '')"
    )
    addr2 = (
        f"COALESCE({target_alias}.{sql_ident(target_address)}, '')"
    )

    addrc1 = (
        f"COALESCE({s1_alias}.{sql_ident(s1_address_compact)}, '')"
    )
    addrc2 = (
        f"COALESCE({target_alias}.{sql_ident(target_address_compact)}, '')"
    )

    country1 = (
        f"COALESCE({s1_alias}.{sql_ident(s1_country)}, '')"
    )
    country2 = (
        f"COALESCE({target_alias}.{sql_ident(target_country)}, '')"
    )

    name_tokens1 = (
        f"COALESCE("
        f"{s1_alias}.{sql_ident(s1_name_tokens)}, "
        f"CAST([] AS VARCHAR[])"
        f")"
    )

    name_tokens2 = (
        f"COALESCE("
        f"{target_alias}.{sql_ident(target_name_tokens)}, "
        f"CAST([] AS VARCHAR[])"
        f")"
    )

    address_tokens1 = (
        f"COALESCE("
        f"{s1_alias}.{sql_ident(s1_address_tokens)}, "
        f"CAST([] AS VARCHAR[])"
        f")"
    )

    address_tokens2 = (
        f"COALESCE("
        f"{target_alias}.{sql_ident(target_address_tokens)}, "
        f"CAST([] AS VARCHAR[])"
        f")"
    )

    # ------------------------------------------------------------
    # ASCII fallback
    # ------------------------------------------------------------

    if s1_name_ascii is not None and target_name_ascii is not None:
        name_ascii1 = (
            f"COALESCE({s1_alias}.{sql_ident(s1_name_ascii)}, '')"
        )
        name_ascii2 = (
            f"COALESCE({target_alias}.{sql_ident(target_name_ascii)}, '')"
        )
    else:
        name_ascii1 = name1
        name_ascii2 = name2

    if s1_address_ascii is not None and target_address_ascii is not None:
        address_ascii1 = (
            f"COALESCE({s1_alias}.{sql_ident(s1_address_ascii)}, '')"
        )
        address_ascii2 = (
            f"COALESCE({target_alias}.{sql_ident(target_address_ascii)}, '')"
        )
    else:
        address_ascii1 = addr1
        address_ascii2 = addr2

    # ------------------------------------------------------------
    # Numeric token expressions
    # ------------------------------------------------------------

    name_numbers1 = (
        f"regexp_extract_all({name1}, '[0-9]+')"
    )

    name_numbers2 = (
        f"regexp_extract_all({name2}, '[0-9]+')"
    )

    address_numbers1 = (
        f"regexp_extract_all({addr1}, '[0-9]+')"
    )

    address_numbers2 = (
        f"regexp_extract_all({addr2}, '[0-9]+')"
    )

    # ------------------------------------------------------------
    # Unique token counts
    # ------------------------------------------------------------

    name_unique1 = f"list_unique({name_tokens1})"
    name_unique2 = f"list_unique({name_tokens2})"

    address_unique1 = f"list_unique({address_tokens1})"
    address_unique2 = f"list_unique({address_tokens2})"

    name_intersection = (
        f"list_intersect({name_tokens1}, {name_tokens2})"
    )

    address_intersection = (
        f"list_intersect({address_tokens1}, {address_tokens2})"
    )

    name_number_intersection = (
        f"list_intersect({name_numbers1}, {name_numbers2})"
    )

    address_number_intersection = (
        f"list_intersect({address_numbers1}, {address_numbers2})"
    )

    # ------------------------------------------------------------
    # Main SELECT
    # ------------------------------------------------------------

    return f"""
        {candidate_alias}.source1_entity_id,

        {candidate_alias}.matched_entity_id,

        {candidate_alias}.matched_source,

        CAST({candidate_alias}.blocking_mask AS SMALLINT)
            AS blocking_mask,

        CAST({candidate_alias}.num_blocking_methods AS TINYINT)
            AS num_blocking_methods,

        {candidate_alias}.blocking_methods,

        -- ========================================================
        -- NAME FEATURES
        -- ========================================================

        CAST(
            CASE
                WHEN {name1} <> ''
                 AND {name1} = {name2}
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS name_exact,

        CAST(
            CASE
                WHEN {namec1} <> ''
                 AND {namec1} = {namec2}
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS name_compact_exact,

        CAST(
            CASE
                WHEN {name_ascii1} <> ''
                 AND {name_ascii1} = {name_ascii2}
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS name_ascii_exact,

        CAST(
            CASE
                WHEN {name1} <> ''
                 AND {name2} <> ''
                THEN jaro_winkler_similarity(
                    {name1},
                    {name2}
                )
                ELSE 0.0
            END
            AS FLOAT
        ) AS name_char_ratio,

        CAST(
            CASE
                WHEN LEAST(
                    {name_unique1},
                    {name_unique2}
                ) = 0
                THEN 0.0

                ELSE
                    CAST(
                        list_unique({name_intersection})
                        AS DOUBLE
                    )
                    /
                    LEAST(
                        {name_unique1},
                        {name_unique2}
                    )
            END
            AS FLOAT
        ) AS name_token_overlap,

        CAST(
            CASE
                WHEN (
                    {name_unique1}
                    +
                    {name_unique2}
                    -
                    list_unique({name_intersection})
                ) = 0
                THEN 0.0

                ELSE
                    CAST(
                        list_unique({name_intersection})
                        AS DOUBLE
                    )
                    /
                    (
                        {name_unique1}
                        +
                        {name_unique2}
                        -
                        list_unique({name_intersection})
                    )
            END
            AS FLOAT
        ) AS name_token_jaccard,

        CAST(
            ABS(
                LENGTH({name1})
                -
                LENGTH({name2})
            )
            AS SMALLINT
        ) AS name_length_diff,

        CAST(
            ABS(
                LENGTH({name_tokens1})
                -
                LENGTH({name_tokens2})
            )
            AS TINYINT
        ) AS name_token_count_diff,

        CAST(
            CASE
                WHEN LEAST(
                    list_unique({name_numbers1}),
                    list_unique({name_numbers2})
                ) = 0
                THEN 0.0

                ELSE
                    CAST(
                        list_unique({name_number_intersection})
                        AS DOUBLE
                    )
                    /
                    LEAST(
                        list_unique({name_numbers1}),
                        list_unique({name_numbers2})
                    )
            END
            AS FLOAT
        ) AS name_numeric_overlap,

        CAST(
            CASE
                WHEN list_unique({name_numbers1}) > 0
                 AND list_unique({name_numbers2}) > 0
                 AND list_unique({name_number_intersection})
                     =
                     list_unique({name_numbers1})
                 AND list_unique({name_number_intersection})
                     =
                     list_unique({name_numbers2})
                THEN 1
                ELSE 0
            END
            AS TINYINT
        ) AS name_numeric_exact,

        CAST(
            CASE
                WHEN GREATEST(
                    LENGTH({name1}),
                    LENGTH({name2})
                ) = 0
                THEN 0.0

                ELSE
                    CAST(
                        LEAST(
                            LENGTH({name1}),
                            LENGTH({name2})
                        )
                        AS DOUBLE
                    )
                    /
                    GREATEST(
                        LENGTH({name1}),
                        LENGTH({name2})
                    )
            END
            AS FLOAT
        ) AS name_length_ratio,

        -- ========================================================
        -- ADDRESS FEATURES
        -- ========================================================

        CAST(
            CASE
                WHEN {addr1} <> ''
                 AND {addr1} = {addr2}
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS address_exact,

        CAST(
            CASE
                WHEN {addrc1} <> ''
                 AND {addrc1} = {addrc2}
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS address_compact_exact,

        CAST(
            CASE
                WHEN {address_ascii1} <> ''
                 AND {address_ascii1} = {address_ascii2}
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS address_ascii_exact,

        CAST(
            CASE
                WHEN {addr1} <> ''
                 AND {addr2} <> ''
                THEN jaro_winkler_similarity(
                    {addr1},
                    {addr2}
                )
                ELSE 0.0
            END
            AS FLOAT
        ) AS address_char_ratio,

        CAST(
            CASE
                WHEN LEAST(
                    {address_unique1},
                    {address_unique2}
                ) = 0
                THEN 0.0

                ELSE
                    CAST(
                        list_unique({address_intersection})
                        AS DOUBLE
                    )
                    /
                    LEAST(
                        {address_unique1},
                        {address_unique2}
                    )
            END
            AS FLOAT
        ) AS address_token_overlap,

        CAST(
            CASE
                WHEN (
                    {address_unique1}
                    +
                    {address_unique2}
                    -
                    list_unique({address_intersection})
                ) = 0
                THEN 0.0

                ELSE
                    CAST(
                        list_unique({address_intersection})
                        AS DOUBLE
                    )
                    /
                    (
                        {address_unique1}
                        +
                        {address_unique2}
                        -
                        list_unique({address_intersection})
                    )
            END
            AS FLOAT
        ) AS address_token_jaccard,

        CAST(
            ABS(
                LENGTH({addr1})
                -
                LENGTH({addr2})
            )
            AS SMALLINT
        ) AS address_length_diff,

        CAST(
            ABS(
                LENGTH({address_tokens1})
                -
                LENGTH({address_tokens2})
            )
            AS TINYINT
        ) AS address_token_count_diff,

        CAST(
            CASE
                WHEN list_unique({address_numbers1}) = 0
                  OR list_unique({address_numbers2}) = 0
                THEN 0

                WHEN list_unique({address_number_intersection})
                     =
                     list_unique({address_numbers1})
                 AND list_unique({address_number_intersection})
                     =
                     list_unique({address_numbers2})
                THEN 1

                ELSE 0
            END
            AS TINYINT
        ) AS address_numeric_exact,

        CAST(
            CASE
                WHEN LEAST(
                    list_unique({address_numbers1}),
                    list_unique({address_numbers2})
                ) = 0
                THEN 0.0

                ELSE
                    CAST(
                        list_unique({address_number_intersection})
                        AS DOUBLE
                    )
                    /
                    LEAST(
                        list_unique({address_numbers1}),
                        list_unique({address_numbers2})
                    )
            END
            AS FLOAT
        ) AS address_numeric_overlap,

        CAST(
            CASE
                WHEN GREATEST(
                    LENGTH({addr1}),
                    LENGTH({addr2})
                ) = 0
                THEN 0.0

                ELSE
                    CAST(
                        LEAST(
                            LENGTH({addr1}),
                            LENGTH({addr2})
                        )
                        AS DOUBLE
                    )
                    /
                    GREATEST(
                        LENGTH({addr1}),
                        LENGTH({addr2})
                    )
            END
            AS FLOAT
        ) AS address_length_ratio,

        -- ========================================================
        -- METADATA
        -- ========================================================

        CAST(
            CASE
                WHEN {country1} <> ''
                 AND {country1} = {country2}
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS country_exact,

        CAST(
            CASE
                WHEN {name1} <> ''
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS name_present,

        CAST(
            CASE
                WHEN {addr1} <> ''
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS address_present,

        -- ========================================================
        -- BLOCKING PROVENANCE
        -- ========================================================

        CAST(
            CASE
                WHEN ({candidate_alias}.blocking_mask & 1) <> 0
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS block_address,

        CAST(
            CASE
                WHEN ({candidate_alias}.blocking_mask & 2) <> 0
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS block_address_compact,

        CAST(
            CASE
                WHEN ({candidate_alias}.blocking_mask & 4) <> 0
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS block_name,

        CAST(
            CASE
                WHEN ({candidate_alias}.blocking_mask & 8) <> 0
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS block_rare_name,

        CAST(
            CASE
                WHEN ({candidate_alias}.blocking_mask & 16) <> 0
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS block_rare_address,

        CAST(
            CASE
                WHEN {candidate_alias}.num_blocking_methods >= 2
                THEN 1 ELSE 0
            END
            AS TINYINT
        ) AS block_hybrid
    """


def feature_column_names() -> list[str]:
    """
    Canonical feature output schema.

    Useful for validation/model code.
    """

    return [
        "source1_entity_id",
        "matched_entity_id",
        "matched_source",
        "blocking_mask",
        "num_blocking_methods",
        "blocking_methods",

        "name_exact",
        "name_compact_exact",
        "name_ascii_exact",
        "name_char_ratio",
        "name_token_overlap",
        "name_token_jaccard",
        "name_length_diff",
        "name_token_count_diff",
        "name_numeric_overlap",
        "name_numeric_exact",
        "name_length_ratio",

        "address_exact",
        "address_compact_exact",
        "address_ascii_exact",
        "address_char_ratio",
        "address_token_overlap",
        "address_token_jaccard",
        "address_length_diff",
        "address_token_count_diff",
        "address_numeric_exact",
        "address_numeric_overlap",
        "address_length_ratio",

        "country_exact",
        "name_present",
        "address_present",

        "block_address",
        "block_address_compact",
        "block_name",
        "block_rare_name",
        "block_rare_address",
        "block_hybrid",
    ]