import sqlite3, pandas as pd

con = sqlite3.connect("fleet_dw.db")
pd.set_option("display.width", 120)
for q in open("olap.sql").read().split("\n\n"):
    print(q.splitlines()[0])
    print(pd.read_sql(q, con).to_string(index=False), "\n")
