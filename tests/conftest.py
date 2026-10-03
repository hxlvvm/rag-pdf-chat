# Chroma needs sqlite3 >= 3.35. Some Linux servers ship an older one; if the optional `pysqlite3-binary`
# package is installed, use its bundled sqlite instead. On normal machines this does nothing.
import sqlite3
import sys

if sqlite3.sqlite_version_info < (3, 35, 0):
    try:
        import pysqlite3

        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        pass
