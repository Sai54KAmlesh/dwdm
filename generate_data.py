import numpy as np, pandas as pd

rng = np.random.default_rng(42)
cities = {"Chennai": ("Tamil Nadu", "South"), "Bengaluru": ("Karnataka", "South"),
          "Hyderabad": ("Telangana", "South"), "Mumbai": ("Maharashtra", "West"),
          "Pune": ("Maharashtra", "West"), "Ahmedabad": ("Gujarat", "West"),
          "Delhi": ("Delhi", "North"), "Jaipur": ("Rajasthan", "North"),
          "Kolkata": ("West Bengal", "East"), "Bhubaneswar": ("Odisha", "East")}
types = {"Truck": 220, "Bus": 180, "Van": 120, "Car": 70}
makes = ["Tata", "Ashok Leyland", "Mahindra", "Eicher", "Maruti", "Toyota"]
groups = [["Brake Pads", "Tyres", "Brake Fluid"], ["Engine Oil", "Oil Filter", "Air Filter"],
          ["Battery", "Alternator", "Starter Motor"], ["Coolant", "Radiator", "Thermostat"],
          ["Clutch Plate", "Gearbox Oil", "Transmission"]]
all_parts = [p for g in groups for p in g]
part_cost = {p: rng.integers(800, 9000) for p in all_parts}
part_cost.update({"Transmission": 38000, "Alternator": 11000, "Radiator": 9500, "Battery": 7500})

rows = []
for v in range(1, 121):
    vtype = rng.choice(list(types), p=[.3, .2, .3, .2])
    city = rng.choice(list(cities))
    year = int(rng.integers(2014, 2023))
    make = rng.choice(makes)
    base_km = types[vtype] * rng.uniform(.7, 1.3)
    odo = (2022 - year) * base_km * 330 + rng.uniform(0, 5000)
    date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 40)))
    gap = 60
    wear = rng.uniform(0, 1)
    while date < pd.Timestamp("2024-12-31"):
        km = base_km * rng.uniform(.8, 1.2)
        odo += km * gap
        age = date.year - year
        idle = max(0, rng.normal(2 + 3 * wear, 1))
        harsh = rng.poisson(1 + 7 * wear)
        temp = rng.normal(84 + 10 * wear + 0.4 * age, 2)
        risk = 0.25 * age / 10 + 0.3 * gap / 120 + 0.2 * harsh / 12 + 0.25 * (temp - 80) / 20 + rng.normal(0, .04)
        need = "High" if risk > .62 else "Medium" if risk > .45 else "Low"
        corrective = rng.random() < min(.9, max(.05, risk - .1))
        if corrective:
            g = groups[rng.choice(len(groups), p=[.25, .15, .2, .25, .15])]
            parts = [p for p in g if rng.random() < .8]
            if rng.random() < .3: parts.append(rng.choice(all_parts))
            parts = sorted(set(parts)) or [g[0]]
        else:
            parts = [p for p, pr in [("Engine Oil", .85), ("Oil Filter", .8), ("Air Filter", .4)] if rng.random() < pr] or ["Engine Oil"]
        cost = sum(part_cost[p] for p in parts) * rng.uniform(1, 1.2) + (1500 if not corrective else 3500)
        down = round(rng.uniform(2, 8) + (rng.uniform(6, 40) if corrective else 0), 1)
        nxt = int(np.clip(round(110 - 85 * risk + rng.normal(0, 6)), 7, 150))
        rows.append(dict(vehicle_id=f"VEH{v:03d}", vehicle_type=vtype, make=make, purchase_year=year,
                         city=city, state=cities[city][0], region=cities[city][1], service_date=date.date(),
                         odometer_km=round(odo), avg_daily_km=round(km, 1), idle_hours=round(idle, 1),
                         harsh_events=harsh, engine_temp_c=round(temp, 1), days_since_last_service=gap,
                         service_type="Corrective" if corrective else "Preventive",
                         parts_replaced="|".join(parts), cost_inr=round(cost), downtime_hours=down,
                         maintenance_need=need, days_to_next_service=nxt))
        gap = nxt
        date += pd.Timedelta(days=nxt)

df = pd.DataFrame(rows)
df.insert(0, "record_id", range(1, len(df) + 1))
df.to_csv("fleet_maintenance.csv", index=False)
print(df.shape, df.maintenance_need.value_counts().to_dict(), df.service_type.value_counts().to_dict())
