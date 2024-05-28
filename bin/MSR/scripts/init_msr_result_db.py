import os
import sqlite3


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('outfn')

    args = ap.parse_args()

    outfn = args.outfn

    # setup output database
    assert not os.path.exists(outfn)

    dbcon = sqlite3.connect(outfn)
    dbcon.executescript("""
    PRAGMA foreign_keys=ON;

    BEGIN TRANSACTION;

    CREATE TABLE "metadata" (key BLOB NOT NULL PRIMARY KEY, value BLOB);
    CREATE UNIQUE INDEX metadata_key_index ON "metadata"("key");

    COMMIT;
    """)

    dbcon.executescript("""
    BEGIN TRANSACTION;

    CREATE TABLE "sampledata" (
        key INTEGER PRIMARY KEY,
        sampletype TEXT NOT NULL,
        limsid VARCHAR NOT NULL,
        indexnr INT NOT NULL,
        barcodenr INT NOT NULL
    );
    CREATE UNIQUE INDEX sampledata_key_index ON "sampledata" (
        "sampletype",
        "limsid",
        "indexnr",
        "barcodenr"
    );

    CREATE TABLE "sampledata_chrom" (
        key INTEGER PRIMARY KEY,
        sampledata_key INTEGER,
        chrom TEXT,
        segment_map_s BLOB,
        tbl_s BLOB
    );
    CREATE UNIQUE INDEX sampledata_chrom_key_index ON "sampledata_chrom" (
        "sampledata_key",
        "chrom"
    );

    COMMIT;
    """)

    dbcon.close()

    return


if __name__ == "__main__":
    main()
