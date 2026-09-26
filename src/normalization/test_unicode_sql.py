from __future__ import annotations

import duckdb


def main() -> None:

    con = duckdb.connect()

    samples = [
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "आदित्य प्रॉपर्टीज एलएलपी",
        "मॉडर्न फाइनेंस",
        "International South Consultants Private Ltd",
        "Moyna's Coffee",
        "LLC Moncada Léarning Center",
        "Pvt. EFS Print Ventures Ltd.",
        "Shri Sai Infratech Co",
        "Fractales Amis Groupe S.A.S",
        "ગુજરાતી બિઝનેસ",
        "বাংলা ব্যবসা",
        "ಕನ್ನಡ ಉದ್ಯಮ",
    ]

    print("\nUNICODE NORMALIZATION TEST")
    print("=" * 90)

    for value in samples:

        result = con.execute(
            """
            SELECT
                ? AS original,

                regexp_replace(
                    lower(trim(?)),
                    '[[:punct:]]',
                    ' ',
                    'g'
                ) AS name_norm,

                regexp_replace(
                    regexp_replace(
                        lower(trim(?)),
                        '[[:punct:]]',
                        ' ',
                        'g'
                    ),
                    '[[:space:][:punct:]]',
                    '',
                    'g'
                ) AS name_compact
            """,
            [value, value, value],
        ).fetchone()

        original, name_norm, name_compact = result

        print(f"\nOriginal      : {original}")
        print(f"Name norm     : {name_norm}")
        print(f"Name compact  : {name_compact}")

    con.close()


if __name__ == "__main__":
    main()