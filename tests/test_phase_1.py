import pandas as pd
import pytest
import numpy as np
import pytest, ast, inspect
import importlib

sp = importlib.import_module("student_project")
from student_project.student_project import load_data, summary_stats, compute_correlations, preprocess_data, train_gb_pipeline, RandomForest

DATA_PATH = "datasets/student-mat-mini.csv"
PY_FORBIDDEN = ["RandomForestClassifier", "DecisionTreeClassifier", "ExtraTreesClassifier",  "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",]

def test_module_is_importable():
    import importlib
    mod = importlib.import_module("student_project.student_project")
    assert hasattr(mod, "preprocess_data")


# ---------- Section A: Data Loading ----------
def test_load_data_type():
    df = load_data()
    assert isinstance(df, pd.DataFrame), "load_data() must return a pandas DataFrame"

def test_load_required_columns():
    df = load_data()
    required = {"G1", "G2", "G3", "absences"}
    assert required.issubset(set(df.columns)), f"Dataset missing required columns: {required - set(df.columns)}"

# ---------- Section A: EDA ----------
def test_summary_stats_type_and_keys():
    stats = summary_stats()
    assert isinstance(stats, dict), "summary_stats() must return a dictionary"
    assert "mean_G3" in stats and "median_absences" in stats, "summary_stats() missing required keys"

def test_compute_correlations_type():
    corr = compute_correlations()
    assert isinstance(corr, pd.DataFrame), "compute_correlations() must return a DataFrame"

# ---------- Section A: Preprocessing ----------
def test_preprocess_output_schema():
    df = load_data()
    processed = preprocess_data(df)
    assert isinstance(processed, pd.DataFrame), "preprocess_data must return a DataFrame"
    # There should be no object-dtype columns after preprocessing
    obj_cols = processed.select_dtypes(include=['object']).columns
    assert len(obj_cols) == 0, f"No object dtypes after preprocessing: found {list(obj_cols)}"
    # No nulls remain
    assert processed.isnull().sum().sum() == 0, "preprocess_data must remove or impute missing values"

    # target variable
    assert "at_risk" in processed.columns, "Must create at_risk = (G3 < 10)"
    assert set(processed["at_risk"].unique()) <= {0,1}, "at_risk must be binary"

def test_preprocess_scaled_numeric():
    df = load_data()
    proc = preprocess_data(df)
    X = proc.drop(columns=["at_risk"])
    num_cols = X.select_dtypes(include=np.number).columns
    assert len(num_cols) > 0, "Expected numeric columns"
    for c in num_cols:
        mn, mx = X[c].min(), X[c].max()
        assert mn >= -1e-6 and mx <= 1+1e-6, f"Column {c} must be scaled to [0,1]"

# ---------- Section B: Gradient Boosting pipeline ----------
def test_gb_pipeline_fits_and_predicts():
    df = load_data()
    processed = preprocess_data(df)
    X = processed.drop(columns=['at_risk'])
    y = processed['at_risk']
    model = train_gb_pipeline(X, y)
    assert model is not None, "train_gb_pipeline must return a (fitted) sklearn-like model/pipeline"
    assert hasattr(model, "predict"), "Returned object must have a predict() method"
    preds = model.predict(X)
    assert len(preds) == len(y)

# ---------- Section C: RandomForest skeleton exists ----------
def test_random_forest_fits_and_predicts():
    df = load_data()
    processed = preprocess_data(df)
    X = processed.drop(columns=['at_risk'])
    y = processed['at_risk']
    rf = RandomForest(n_estimators=3, max_depth=3, sample_size=min(32, len(X)))
    # fit may be NotImplemented at first; students must implement
    try:
        rf.fit(X.values, y.values)
    except NotImplementedError:
        pytest.skip("RandomForest.fit not implemented yet")
    preds = rf.predict(X.values)
    assert len(preds) == len(y), "predict() must return one label per row"

# ---------- Extra test A: dataset contents ----------
def test_load_data_columns():
    df = load_data()
    expected = {"school", "sex", "age", "address", "G1", "G2", "G3"}
    assert expected.issubset(set(df.columns)), "Extra: dataset missing expected columns"

# ---------- Extra test B: summary stats reasonable ----------
def test_summary_stats_ranges():
    df = load_data()
    mean_G3 = df["G3"].mean()
    median_absences = df["absences"].median()
    assert 0 <= mean_G3 <= 20, "Extra: mean_G3 out of valid range"
    assert median_absences >= 0, "Extra: median_absences negative"

# ---------- Extra test C: preprocessing scaling on 'age' ----------
def test_preprocess_scaling():
    df = load_data()
    proc = preprocess_data(df)
    # Extra test assumes students scaled numeric columns between 0 and 1
    numeric_cols = proc.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        assert proc[col].min() >= 0 and proc[col].max() <= 1, f"Extra: numeric column '{col}' must be scaled to [0,1]"

# ---------- Extra test D: GB pipeline includes classifier ----------
def test_gb_pipeline_has_classifier():
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingClassifier

    df = load_data()
    processed = preprocess_data(df)
    X = processed.drop(columns=['at_risk'])
    y = processed['at_risk']
    model = train_gb_pipeline(X, y)
    assert isinstance(model, Pipeline), "Extra: train_gb_pipeline must return an sklearn Pipeline"
    assert hasattr(model, "named_steps"), "Extra: pipeline must expose named_steps"
    assert "classifier" in model.named_steps, "Extra: pipeline missing 'classifier' step"
    assert model.steps[-1][0] == "classifier", "Extra: classifier must be the final pipeline step"
    classifier = model.named_steps["classifier"]
    assert isinstance(classifier, GradientBoostingClassifier), \
        "Extra: classifier step must be a GradientBoostingClassifier instance"


# --- Policy: ban forbidden sklearn APIs in student_project.py ---
def test_policy_no_forbidden_apis():
    src = inspect.getsource(sp)
    tree = ast.parse(src)
    names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    for bad in PY_FORBIDDEN:
        assert bad not in names, f"Forbidden API detected: {bad}"

# --- A. Target & leakage ---
def test_target_is_hardcoded_and_correct():
    df = load_data()
    proc = preprocess_data(df)
    assert "at_risk" in proc.columns
    expected = (df["G3"] < 10).astype(int).values
    actual = proc["at_risk"].values
    assert np.array_equal(actual, expected), "at_risk must be (G3 < 10)"
    # leakage: ensure raw G1/G2/G3 not in features
    feat_cols = [c for c in proc.columns if c != "at_risk"]
    bad = {"G1","G2","G3"}
    assert not (set(feat_cols) & bad), "Remove grade columns from feature matrix"

# --- B. Scaling & types ---
def test_no_objects_and_scaled_to_unit():
    df = load_data()
    proc = preprocess_data(df)
    X = proc.drop(columns=["at_risk"])
    assert not any(X[c].dtype == "O" for c in X.columns), "No object dtypes allowed"
    for c in X.select_dtypes(include=np.number).columns:
        mn, mx = X[c].min(), X[c].max()
        assert mn >= -1e-6 and mx <= 1+1e-6, f"{c} must be in [0,1]"

# --- C. GB pipeline: structure + metrics ---
def test_gb_structure_and_metric_gate():
    from sklearn.model_selection import train_test_split
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    df = load_data()
    proc = preprocess_data(df)
    X = proc.drop(columns=["at_risk"])
    y = proc["at_risk"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = train_gb_pipeline(Xtr, ytr)
    # structure (if pipeline)
    if hasattr(model, "named_steps"):
        names = set(model.named_steps.keys())
        assert "classifier" in names, "Pipeline missing classifier"
        assert "preprocessor" in names, "Pipeline should include preprocessor"

    yp = model.predict(Xte)
    f1 = f1_score(yte, yp, zero_division=0)

    # dummy baseline
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    f1_dummy = f1_score(yte, dummy.predict(Xte), zero_division=0)

    # auc if available
    try:
        auc = roc_auc_score(yte, getattr(model, "predict_proba")(Xte)[:,1])
    except Exception:
        auc = 0.0

    assert (f1 >= 0.40) or (f1 >= f1_dummy + 0.10) or (auc >= 0.50), \
        f"GB failed metric gate: f1={f1:.2f}, dummy={f1_dummy:.2f}, auc={auc:.2f}"

# --- D. RF structure + sanity metrics ---
def test_rf_structure_and_sanity_metric():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    df = load_data()
    proc = preprocess_data(df)
    X = proc.drop(columns=["at_risk"]).values
    y = proc["at_risk"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    rf = RandomForest(n_estimators=5, max_depth=4, sample_size=min(64, len(Xtr)), random_state=42)
    try:
        rf.fit(Xtr, ytr)
    except NotImplementedError:
        pytest.fail("RandomForest.fit must be implemented for Phase 1")

    assert hasattr(rf, "trees") and len(rf.trees) == rf.n_estimators, "RF must populate trees"

    yp = rf.predict(Xte)
    assert len(yp) == len(yte), "predict must match input rows"

    # Light gate: either clear floor or close to GB (if GB is good)
    f1_rf = f1_score(yte, yp, zero_division=0)

    # Optional: compare to GB on same split
    gb = train_gb_pipeline(pd.DataFrame(Xtr), pd.Series(ytr))
    f1_gb = f1_score(yte, gb.predict(pd.DataFrame(Xte)), zero_division=0)

    assert (f1_rf >= 0.30) or (f1_rf >= (f1_gb - 0.15)), \
        f"RF metric gate failed: f1_rf={f1_rf:.2f}, f1_gb={f1_gb:.2f}"