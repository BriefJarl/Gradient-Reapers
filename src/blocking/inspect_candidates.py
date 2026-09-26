import duckdb

PATH = "artifacts/blocking/union/train_exact_rare_address_union.parquet"

con = duckdb.connect()

print("=" * 80)
print("CANDIDATE SCHEMA")
print("=" * 80)

print(
    con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{PATH}')"
    ).fetchdf().to_string(index=False)
)

print("\n" + "=" * 80)
print("SAMPLE")
print("=" * 80)

print(
    con.execute(
        f"SELECT * FROM read_parquet('{PATH}') LIMIT 5"
    ).fetchdf().to_string(index=False)
)