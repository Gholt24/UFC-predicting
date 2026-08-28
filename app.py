import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock Dataset Baseline
UFC_DATA_URL = "https://ufcapi.aristotle.me/api/fighters"


def load_all_fighters():
    try:
        res = requests.get(f"{UFC_DATA_URL}?limit=5000", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return [
        {
            "id": "1",
            "name": "Umar Nurmagomedov",
            "weight_class": "Bantamweight",
            "age": 30,
            "reach": 69,
            "slpm": 4.75,
            "sapm": 1.15,
            "tdd_pct": 0.85,
            "td_avg": 4.10,
            "stance": "Switch",
        },
        {
            "id": "2",
            "name": "Song Yadong",
            "weight_class": "Bantamweight",
            "age": 28,
            "reach": 67,
            "slpm": 4.38,
            "sapm": 3.70,
            "tdd_pct": 0.73,
            "td_avg": 0.45,
            "stance": "Orthodox",
        },
        {
            "id": "3",
            "name": "Sean Woodson",
            "weight_class": "Featherweight",
            "age": 34,
            "reach": 78,
            "slpm": 5.41,
            "sapm": 4.01,
            "tdd_pct": 0.83,
            "td_avg": 0.00,
            "stance": "Switch",
        },
        {
            "id": "4",
            "name": "Su Mudaerji",
            "weight_class": "Flyweight",
            "age": 30,
            "reach": 72,
            "slpm": 4.49,
            "sapm": 2.76,
            "tdd_pct": 0.66,
            "td_avg": 0.40,
            "stance": "Southpaw",
        },
    ]


FIGHTERS_DB = load_all_fighters()

scaler = StandardScaler()
X_dummy = np.array(
    [
        [2, 5, 2.1, 1.5, 0, 1],
        [-4, -3, -1.5, -0.8, 1, 0],
        [0, 2, 0.8, 0.2, 0, 0],
        [-6, 6, 2.5, 2.0, 1, 2],
    ]
)
scaler.fit(X_dummy)
model = LogisticRegression()
model.fit(scaler.transform(X_dummy), np.array([1, 0, 1, 1]))


@app.get("/api/fighters")
def search_fighters(
    query: str = Query(""), weight_class: str = Query("All")
):
    results = []
    for f in FIGHTERS_DB:
        match_q = query.lower() in f.get("name", "").lower()
        match_w = (
            True
            if weight_class == "All"
            else f.get("weight_class") == weight_class
        )
        if match_q and match_w:
            results.append(f)
    return results[:50]


@app.get("/api/predict")
def predict_matchup(fighter1_id: str, fighter2_id: str):
    f1 = next((f for f in FIGHTERS_DB if str(f.get("id")) == fighter1_id), None)
    f2 = next((f for f in FIGHTERS_DB if str(f.get("id")) == fighter2_id), None)
    if not f1 or not f2:
        raise HTTPException(status_code=404, detail="Fighter not found")

    age_diff = f1.get("age", 30) - f2.get("age", 30)
    reach_diff = f1.get("reach", 70) - f2.get("reach", 70)
    striking_diff = (f1.get("slpm", 3.5) - f1.get("sapm", 3.0)) - (
        f2.get("slpm", 3.5) - f2.get("sapm", 3.0)
    )
    grap_edge = (f1.get("td_avg", 1.0) * (1 - f2.get("tdd_pct", 0.6))) - (
        f2.get("td_avg", 1.0) * (1 - f1.get("tdd_pct", 0.6))
    )

    st1, st2 = f1.get("stance", "Orthodox"), f2.get("stance", "Orthodox")
    is_open = (
        1
        if (st1 == "Southpaw" and st2 == "Orthodox")
        or (st1 == "Orthodox" and st2 == "Southpaw")
        else 0
    )
    st_diff = (2 if st1 == "Switch" else (1 if st1 == "Southpaw" else 0)) - (
        2 if st2 == "Switch" else (1 if st2 == "Southpaw" else 0)
    )

    feats = scaler.transform([[
        age_diff,
        reach_diff,
        striking_diff,
        grap_edge,
        is_open,
        st_diff,
    ]])
    prob1 = float(model.predict_proba(feats)[0][1])

    return {
        "fighter1": f1["name"],
        "fighter2": f2["name"],
        "fighter1_prob": f"{round(prob1 * 100, 1)}%",
        "fighter2_prob": f"{round((1 - prob1) * 100, 1)}%",
    }
