"""Task 3 - Supervised and Unsupervised Models."""
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans
from loan_data import make_loan_data

def main():
    df = make_loan_data()
    target = "default"
    X = df.drop(columns=[target, "application_date"])
    y = df[target]
    X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=.4,random_state=42,stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=.5,random_state=42,stratify=y_temp)
    num = ["age","income","employment_years","credit_score","loan_amount","debt_to_income"]
    cat = ["employment_type","education","purpose","grade"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num),
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, oob_score=True, bootstrap=True, random_state=42, n_jobs=-1)
    }
    results, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess",prep),("model",model)])
        pipe.fit(X_train,y_train)
        pred = pipe.predict(X_val)
        score = accuracy_score(y_val,pred)
        results.append((name,score)); fitted[name]=pipe
        print(f"{name} validation accuracy: {score:.3f}")
        if name == "RandomForest": print(f"RandomForest OOB score: {model.oob_score_:.3f}")

    # XGBoost classifier with validation-set early stopping.
    try:
        from xgboost import XGBClassifier
        xgb_prep = prep
        Xt = xgb_prep.fit_transform(X_train, y_train)
        Xv = xgb_prep.transform(X_val)
        xgb = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=.05,
            subsample=.9, colsample_bytree=.9, eval_metric="logloss",
            early_stopping_rounds=20, random_state=42, n_jobs=-1
        )
        xgb.fit(Xt, y_train, eval_set=[(Xv,y_val)], verbose=False)
        xgb_pred = xgb.predict(Xv)
        xgb_score = accuracy_score(y_val,xgb_pred)
        results.append(("XGBoost",xgb_score))
        fitted["XGBoost"] = xgb
        print(f"XGBoost validation accuracy (early stopping): {xgb_score:.3f}")
        try:
            names = xgb_prep.get_feature_names_out()
            imp = pd.Series(xgb.feature_importances_,index=names).sort_values(ascending=False).head(5)
            print("XGBoost top 5 features:\n",imp.to_string())
        except Exception: print("XGBoost top 5 feature importances:", np.sort(xgb.feature_importances_)[-5:][::-1])
    except ImportError:
        print("XGBoost unavailable; install with: py -m pip install xgboost")

    # Logistic predict_proba for 5 validation examples and compare hard predictions.
    log_pipe = fitted["LogisticRegression"]
    print("\nLogisticRegression predict_proba for 5 rows:")
    print(np.round(log_pipe.predict_proba(X_val.head(5)),3))
    print("predict:", log_pipe.predict(X_val.head(5)).tolist())
    rf = fitted["RandomForest"].named_steps["model"]
    feature_names = fitted["RandomForest"].named_steps["preprocess"].get_feature_names_out()
    top_rf = pd.Series(rf.feature_importances_,index=feature_names).sort_values(ascending=False).head(5)
    print("\nRandom Forest top 5 features:\n",top_rf.to_string())

    # Regression on loan amount (target is generated loan amount itself); use numeric predictors excluding loan_amount.
    reg_features = ["age","income","employment_years","credit_score","debt_to_income"]
    Xr = df[reg_features].copy()
    yr = df["loan_amount"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr,yr,test_size=.2,random_state=42)
    Xr_train = Xr_train.fillna(Xr_train.median()); Xr_test = Xr_test.fillna(Xr_train.median())
    print("\nRegression coefficient comparison (features: age, income, employment_years, credit_score, debt_to_income)")
    for name, reg in [("LinearRegression",LinearRegression()),("Ridge",Ridge(alpha=1.0)),("Lasso",Lasso(alpha=10.0,max_iter=10000))]:
        reg.fit(Xr_train,yr_train)
        print(f"{name} coefficients:", np.round(reg.coef_,3).tolist())
        if name=="Lasso": print("Lasso zero coefficients:", int(np.sum(np.isclose(reg.coef_,0))))

    # KMeans clustering on 3 numeric columns, k=2..8.
    cluster_data = df[["income","loan_amount","credit_score"]].copy()
    cluster_data = cluster_data.fillna(cluster_data.median())
    cluster_scaled = StandardScaler().fit_transform(cluster_data)
    inertias, silhouettes = [], []
    for k in range(2,9):
        km = KMeans(n_clusters=k,random_state=42,n_init=10)
        labels = km.fit_predict(cluster_scaled)
        inertias.append(km.inertia_); silhouettes.append(silhouette_score(cluster_scaled,labels))
        print(f"KMeans k={k}: inertia={km.inertia_:.2f}, silhouette={silhouettes[-1]:.3f}")
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(range(2,9),inertias,marker="o"); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
    ax1.set_title("K-Means Elbow Curve")
    fig.tight_layout(); fig.savefig("kmeans_elbow.png",dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(2,9),silhouettes,marker="o"); ax.set_xlabel("k"); ax.set_ylabel("Silhouette score")
    ax.set_title("K-Means Silhouette Scores")
    fig.tight_layout(); fig.savefig("kmeans_silhouette.png",dpi=150); plt.close(fig)
    best_k = list(range(2,9))[int(np.argmax(silhouettes))]
    print(f"\nBest silhouette among tested k: {best_k}")
    print("Segments are groups with similar income, requested loan amount and credit score; inspect cluster centres to profile them.")
    print("\nValidation results table")
    print(pd.DataFrame(results,columns=["Model","Validation score"]).to_string(index=False))
    print("Plots saved: kmeans_elbow.png, kmeans_silhouette.png")

if __name__ == "__main__": main()
