import sqlite3, pandas as pd

df = pd.read_csv("fleet_maintenance.csv", parse_dates=["service_date"]).drop_duplicates().dropna(subset=["record_id"])
con = sqlite3.connect("fleet_dw.db")
con.executescript(open("schema.sql").read())

d = pd.DataFrame({"full_date": df.service_date.sort_values().unique()})
d["time_key"] = d.full_date.dt.strftime("%Y%m%d").astype(int)
d["day"], d["month"], d["month_name"] = d.full_date.dt.day, d.full_date.dt.month, d.full_date.dt.strftime("%b")
d["quarter"], d["year"] = d.full_date.dt.quarter, d.full_date.dt.year
dim_time = d.assign(full_date=d.full_date.dt.strftime("%Y-%m-%d"))

def dim(cols, key):
    t = df[cols].drop_duplicates().reset_index(drop=True)
    t.insert(0, key, t.index + 1)
    return t

dim_vehicle = dim(["vehicle_id", "vehicle_type", "make", "purchase_year"], "vehicle_key")
dim_location = dim(["city", "state", "region"], "location_key")
dim_service = dim(["service_type"], "service_type_key")
parts = sorted({p for s in df.parts_replaced for p in s.split("|")})
dim_part = pd.DataFrame({"part_key": range(1, len(parts) + 1), "part_name": parts})

f = (df.merge(dim_vehicle, on=["vehicle_id", "vehicle_type", "make", "purchase_year"])
       .merge(dim_location, on=["city", "state", "region"]).merge(dim_service, on="service_type"))
f["time_key"] = f.service_date.dt.strftime("%Y%m%d").astype(int)
f["parts_count"] = f.parts_replaced.str.count(r"\|") + 1
f = f.rename(columns={"record_id": "service_key"})
fact = f[["service_key", "time_key", "vehicle_key", "location_key", "service_type_key", "odometer_km",
          "avg_daily_km", "idle_hours", "harsh_events", "engine_temp_c", "days_since_last_service",
          "days_to_next_service", "parts_count", "cost_inr", "downtime_hours", "maintenance_need"]]

pk = dict(zip(dim_part.part_name, dim_part.part_key))
bridge = pd.DataFrame([(r.service_key, pk[p]) for r in f.itertuples() for p in r.parts_replaced.split("|")],
                      columns=["service_key", "part_key"])

for name, t in [("dim_time", dim_time), ("dim_vehicle", dim_vehicle), ("dim_location", dim_location),
                ("dim_service_type", dim_service), ("dim_part", dim_part), ("fact_service", fact),
                ("bridge_service_part", bridge)]:
    t.to_sql(name, con, if_exists="append", index=False)
    print(name, len(t))
con.commit()
