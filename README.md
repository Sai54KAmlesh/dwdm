# Fleet Management: DW and Data Mining Lab Project

Contents
- DWDM_Fleet_Management_Project.pptx   the presentation (follows the lab template)
- PRESENTATION_GUIDE.md                plain-language notes: problem, roles, results, viva answers
- Code: generate_data.py, schema.sql, build_dw.py, olap.sql, olap.py, mining.py, dashboard.py, run_all.py

Run
    pip install -r requirements.txt
    python run_all.py               # builds data, warehouse, OLAP, mining, then opens the dashboard
    python run_all.py --no-dashboard
    streamlit run dashboard.py      # dashboard only

Dashboard tabs: Overview, Classification, Clustering, Association rules, Prediction, Watchlist.
Every setting (k, features, model, depth, support, confidence, lift, test share) recomputes live.
Default settings reproduce the mining.py and slide numbers exactly.
The dataset is synthetic.
