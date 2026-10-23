# Presentation Guide: Fleet Management Analytics

## 1. The problem (30 seconds)
A fleet company spends a lot on repairs and suffers sudden breakdowns, but its records are just a long list of service visits. We turn that list into answers: where the money goes, which vehicles need service soon, and which parts fail together.

Opening line: "Fleet companies spend heavily on repairs and breakdowns are costly. We built a data warehouse so managers can see where the cost is, then used data mining to find which vehicles need service soon, which wear out fastest, and which parts fail together. Everything is shown on a live dashboard. The data is simulated, but the method works the same on real fleet data."

## 2. The solution in two steps
1. Data warehouse: organise the 2,028 service records into a star schema (one fact table, four dimensions, one bridge for parts) so questions are fast to ask. OLAP operations: roll-up (zoom out), drill-down (zoom in), slice (one value), dice (a block), pivot (flip axes).
2. Data mining: four tasks (below), then one dashboard on top.

## 3. Implementation steps
| Step | File | What it does |
|---|---|---|
| 1 | generate_data.py | Creates the synthetic dataset (2,028 rows, 120 vehicles, 2022 to 2024) |
| 2 | schema.sql, build_dw.py | ETL into the SQLite star schema |
| 3 | olap.sql, olap.py | The five OLAP queries |
| 4 | mining.py | Classification, clustering, rules, prediction |
| 5 | dashboard.py | Live dashboard with adjustable settings |

Run everything with `python run_all.py`.

## 4. Roles for two people
Person A, data and warehouse: dataset and attributes (slides 2 to 4), warehouse design and ETL (5 to 9), OLAP (7 and 10), dashboard and demo (16 and 17). Files: generate_data.py, schema.sql, build_dw.py, olap.sql, dashboard.py.
Person B, data mining: classification, clustering, rules, prediction (slides 11 to 15) and the conclusion (19). File: mining.py.
Both should be able to explain the other's part in one sentence.

## 5. Mining results in simple words
Classification (Low, Medium or High need): inputs are age, odometer, daily km, idle hours, harsh events, engine temperature, days since last service and vehicle type. 75/25 split. Decision Tree 74%, Random Forest 81% (200 trees voting). High-need recall 63%, and misses are never rated Low. Top inputs: engine temperature, harsh events, age, odometer.

Clustering: K-Means on 120 vehicles using six averages (daily km, idle, harsh events, temperature, age, repair share), standardised. k = 4 has the best silhouette score of the k values tested (3 to 6), at 0.258. Groups: Low, Moderate, High and Extreme wear. Extreme wear vehicles brake harshly, run hot, have 55% repair visits and cost about INR 4.0 lakh each against 2.4 lakh for Low wear.

Prediction (days to next service): same inputs. Linear Regression beat Random Forest. Average error 5.5 days, R2 0.59.

Association rules: Apriori with min support 0.04, confidence 0.6, lift above 1.5 gives 36 rules. Families that fail together: electrical (Alternator, Battery, Starter Motor), cooling (Coolant, Radiator, Thermostat), braking (Brake Pads, Tyres, Brake Fluid).

## 6. Support, confidence, lift
Each service visit is a basket and each part replaced is an item. Example rule: {Alternator, Battery} -> Starter Motor.
- Support = visits with all the parts / all visits (0.05, about 100 visits).
- Confidence = visits with all parts / visits with the "if" parts (0.84).
- Lift = confidence / how often the "then" part appears overall (11, so eleven times more than chance).
Support is low because each breakdown family is rare among all visits (only 42% of visits are repairs, spread over 5 families). A rare rule can still be strong. With 15 parts a high minimum support (0.20) would give zero rules.

## 7. Why these models and not others
Only two models per task were tested; the comparison with others is reasoning.
- Random Forest over Logistic Regression, KNN, SVM, Naive Bayes, neural nets: handles mixed inputs and threshold patterns, resists over-fitting, ranks input importance, needs little tuning. XGBoost is the natural next step.
- K-Means over hierarchical and DBSCAN: simple, fast, gives every vehicle one group, easy to describe by averages.
- Linear Regression over Ridge, SVR, boosting: days fall roughly in a straight line with wear, so the simple model wins and is easiest to explain.
- Apriori over FP-Growth: same rules; FP-Growth is faster on big data, Apriori is simpler to explain.

## 8. Questions to be ready for
- Why is accuracy so high? The data is generated with patterns built in. Real data scores lower.
- Why is R2 only 0.59? Random noise was added on purpose, as in real life.
- Why does Random Forest win on classification but not prediction? Classes depend on thresholds; day counts change smoothly.
- Why mirrored rules? Both directions share support and lift; only confidence changes.
- Did you try XGBoost or SVM? No. Testing them is the next step.

## 9. Dashboard demo flow
Overview: drill the time level from Year to Month and change the breakdown. Clustering: move the k slider and watch the groups and silhouette change. Rules: lower min support and watch rules appear. Prediction: switch the model and use the Try a vehicle sliders. Watchlist: see which vehicles are due soon.
