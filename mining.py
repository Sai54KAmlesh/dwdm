import sqlite3, json, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

con = sqlite3.connect("fleet_dw.db")
df = pd.read_sql("""SELECT f.*, t.full_date, t.year, v.vehicle_id, v.vehicle_type, v.purchase_year, l.region, s.service_type
    FROM fact_service f JOIN dim_time t USING(time_key) JOIN dim_vehicle v USING(vehicle_key)
    JOIN dim_location l USING(location_key) JOIN dim_service_type s USING(service_type_key)""", con)
df["age"] = df.year - df.purchase_year
X = pd.get_dummies(df[["age", "odometer_km", "avg_daily_km", "idle_hours", "harsh_events", "engine_temp_c",
                       "days_since_last_service", "vehicle_type"]])
res = {}

# 1. Classification: maintenance need
Xtr, Xte, ytr, yte = train_test_split(X, df.maintenance_need, test_size=.25, random_state=1, stratify=df.maintenance_need)
models = {"Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=1),
          "Random Forest": RandomForestClassifier(200, random_state=1)}
res["classification"] = {}
for n, m in models.items():
    p = m.fit(Xtr, ytr).predict(Xte)
    res["classification"][n] = {"accuracy": round(accuracy_score(yte, p), 3), "f1_macro": round(f1_score(yte, p, average="macro"), 3)}
rf = models["Random Forest"]
labels = ["Low", "Medium", "High"]
cm = confusion_matrix(yte, rf.predict(Xte), labels=labels)
res["confusion"] = cm.tolist()
res["importance"] = dict(pd.Series(rf.feature_importances_, X.columns).nlargest(6).round(3))
df["need_pred"] = rf.predict(X)

fig, ax = plt.subplots(figsize=(4, 3.4))
ax.imshow(cm, cmap="Greys")
ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig("figs/confusion.png", dpi=200); plt.close()

# 2. Clustering: vehicles by usage / wear
v = df.groupby("vehicle_id").agg(avg_daily_km=("avg_daily_km", "mean"), idle_hours=("idle_hours", "mean"),
    harsh_events=("harsh_events", "mean"), engine_temp_c=("engine_temp_c", "mean"), age=("age", "mean"),
    corrective_rate=("service_type", lambda s: (s == "Corrective").mean()), total_cost=("cost_inr", "sum"))
Z = StandardScaler().fit_transform(v.drop(columns="total_cost"))
sil = {k: round(silhouette_score(Z, KMeans(k, n_init=10, random_state=1).fit_predict(Z)), 3) for k in range(3, 7)}
k = max(sil, key=sil.get)
v["cluster"] = KMeans(k, n_init=10, random_state=1).fit_predict(Z)
order = v.groupby("cluster").corrective_rate.mean().sort_values().index
names = dict(zip(order, ["Low wear", "Moderate wear", "High wear", "Extreme wear"][:k]))
v["cluster_name"] = v.cluster.map(names)
res["clustering"] = {"silhouette": sil, "k": int(k), "profile": v.groupby("cluster_name").mean(numeric_only=True).round(2).drop(columns="cluster").to_dict("index"),
                     "sizes": v.cluster_name.value_counts().to_dict()}
fig, ax = plt.subplots(figsize=(5, 3.4))
for n, c in zip(["Low wear", "Moderate wear", "High wear", "Extreme wear"][:k], ["#9a9a9a", "#555555", "#111111", "#c0392b"]):
    s = v[v.cluster_name == n]; ax.scatter(s.harsh_events, s.engine_temp_c, label=n, color=c, s=28)
ax.set_xlabel("Avg harsh events per service"); ax.set_ylabel("Avg engine temp (C)"); ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig("figs/clusters.png", dpi=200); plt.close()

# 3. Association rules: part co-occurrence
bk = df[["service_key"]].merge(pd.read_sql("SELECT service_key, part_name FROM bridge_service_part JOIN dim_part USING(part_key)", con))
basket = pd.crosstab(bk.service_key, bk.part_name).astype(bool)
fi = apriori(basket, min_support=.04, use_colnames=True)
rules = association_rules(fi, metric="confidence", min_threshold=.6).query("lift > 1.5").sort_values("lift", ascending=False)
rules["antecedents"] = rules.antecedents.map(lambda s: ", ".join(sorted(s)))
rules["consequents"] = rules.consequents.map(lambda s: ", ".join(sorted(s)))
rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)
rules.to_csv("rules.csv", index=False)
res["rules"] = {"n": len(rules), "top": rules.head(6).to_dict("records")}

# 4. Prediction: days to next service
Xtr, Xte, ytr, yte = train_test_split(X, df.days_to_next_service, test_size=.25, random_state=1)
res["prediction"] = {}
reg = {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(200, random_state=1)}
for n, mod in reg.items():
    p = mod.fit(Xtr, ytr).predict(Xte)
    res["prediction"][n] = {"MAE_days": round(mean_absolute_error(yte, p), 2), "R2": round(r2_score(yte, p), 3)}
best = min(res["prediction"], key=lambda n: res["prediction"][n]["MAE_days"])
m = reg[best]; res["prediction"]["best"] = best
pred = m.predict(Xte)
fig, ax = plt.subplots(figsize=(4.4, 3.4))
ax.scatter(yte, pred, s=8, color="#333333", alpha=.5); ax.plot([0, 150], [0, 150], color="#c0392b", lw=1)
ax.set_xlabel("Actual days to next service"); ax.set_ylabel("Predicted"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figs/prediction.png", dpi=200); plt.close()
df["days_pred"] = m.predict(X)

last = df.sort_values("full_date").groupby("vehicle_id").tail(1).set_index("vehicle_id")
v["last_service"] = last.full_date
v["need_now"] = last.need_pred
v["days_to_next"] = last.days_pred.round()
v["next_service_due"] = (pd.to_datetime(v.last_service) + pd.to_timedelta(v.days_to_next, unit="D")).dt.date
v.to_csv("vehicle_summary.csv")
json.dump(res, open("results.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
