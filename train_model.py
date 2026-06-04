import os, re, json, pickle, warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from collections import Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR,"tweet_emotions.csv")
MODEL_DIR = os.path.join(BASE_DIR,"models")
os.makedirs(MODEL_DIR,exist_ok=True)

MODEL_PKL = os.path.join(MODEL_DIR, "mood_classifier.pkl")
REPORT_JSON = os.path.join(MODEL_DIR, "training_report.json")
CM_PNG  = os.path.join(MODEL_DIR, "confusion_matrix.png")
ACC_PNG = os.path.join(MODEL_DIR, "accuracy_comparison.png")

LABEL_MAP = {
    "happiness" : "happy",
    "love" : "happy",
    "enthusiasm": "happy",
    "fun" : "happy",
    "relief" : "happy",
    "surprise" : "happy",
    "neutral" : "neutral",
    "boredom": "neutral",
    "empty" : "neutral",
    "sadness" : "sad",
    "worry" : "anxious",
    "hate" : "angry",
    "anger" : "angry",
}
MOOD_LABELS = ["happy", "neutral", "sad", "anxious", "angry"]


def load_data():
    if not os.path.exists(DATASET):
        raise FileNotFoundError(
            f"\n[ERROR] tweet_emotions.csv not found at:\n  {DATASET}\n"
            "Add tweet_emotions.csv in the same folder as train_model.py"
        )
    df = pd.read_csv(DATASET)
    df = df.dropna(subset=["sentiment","content"])
    df["mood"] = df["sentiment"].map(LABEL_MAP)
    df = df.dropna(subset=["mood"])
    return df


def eda(df):
    df["text_len"] = df["content"].str.len()
    return df


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+","", str(text))
    text = re.sub(r"@\w+","",  text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-zA-Z\s!?]", " ", text)
    text = text.lower().strip()
    text = re.sub(r"\s+"," ", text)
    return text


def preprocess(df):
    df["clean"] = df["content"].apply(clean_text)
    df = df[df["clean"].str.strip().str.len() > 2]
    return df


def make_tfidf():
    return TfidfVectorizer(
        ngram_range = (1, 2),
        max_features = 40_000,
        sublinear_tf  = True,
        min_df = 2,
        strip_accents = "unicode",
    )


def get_classifiers():
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", LogisticRegression(
                C = 5,max_iter = 1000,class_weight = "balanced",solver = "lbfgs",
            )),
        ]),

        "Linear SVM": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, class_weight="balanced"),
                cv=3,
            )),
        ]),

        "Decision Tree": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", DecisionTreeClassifier(
                max_depth = 30,class_weight = "balanced",random_state = 42,
            )),
        ]),

        "k-NN (k=5)": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range  = (1, 1),max_features = 15_000,sublinear_tf = True,
            )),
            ("clf",KNeighborsClassifier(
                n_neighbors = 5, metric = "cosine",n_jobs  = -1,
            )),
        ]),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = get_classifiers()
    all_results = {}
    best_name  = None
    best_model  = None
    best_acc = 0.0

    for name, pipeline in models.items():
        print(f"Training: {name}")

        t0 = datetime.now()
        pipeline.fit(X_train, y_train)
        elapsed = (datetime.now() - t0).total_seconds()

        preds  = pipeline.predict(X_test)
        acc    = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds,target_names = MOOD_LABELS,output_dict  = True,zero_division = 0,)
        cm    = confusion_matrix(y_test, preds, labels=MOOD_LABELS)
        macro = report["macro avg"]

        cv_mean, cv_std = 0.0, 0.0
        if name in ("Logistic Regression", "Linear SVM"):
            cv = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
            cv_mean, cv_std = cv.mean(), cv.std()
        else:
            cv_mean = acc

        all_results[name] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro["f1-score"]), 4),
            "macro_prec": round(float(macro["precision"]), 4),
            "macro_recall": round(float(macro["recall"]), 4),
            "cv_mean" : round(float(cv_mean), 4),
            "cv_std" : round(float(cv_std), 4),
            "train_time_s": round(float(elapsed), 2),
            "confusion_matrix": cm.tolist(),
            "per_class" : {m: report[m] for m in MOOD_LABELS},
        }

        print(f"Done, accuracy: {acc*100:.2f}%  macro F1: {macro['f1-score']*100:.2f}%")

        if acc > best_acc:
            best_acc  = acc
            best_name = name
            best_model = pipeline

    return all_results,best_name,best_model


def streak_search(mood_sequence: list) -> dict:
    NEGATIVE = {"sad", "anxious", "angry"}
    best = {"length": 0, "start": -1, "path": []}
    i = 0
    while i < len(mood_sequence):
        if mood_sequence[i] in NEGATIVE:
            chain = []
            j = i
            while j < len(mood_sequence) and mood_sequence[j] in NEGATIVE:
                chain.append(mood_sequence[j])
                j += 1
            if len(chain) > best["length"]:
                best = {"length": len(chain), "start": i, "path": chain[:]}
            i = j
        else:
            i += 1
    return best


def save_plots(results, best_name, df):
    plt.style.use("dark_background")
    BAR_COLORS = ["#6ee7b7", "#93c5fd", "#fca5a5", "#f9a8d4"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d0f14")
    fig.suptitle("Smart Journal: Model Performance Comparison", color="white", fontsize=15, fontweight="bold", y=1.02)

    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    f1s = [results[n]["macro_f1"] * 100 for n in names]
    cvs = [results[n]["cv_mean"]  * 100 for n in names]

    for ax, vals, title, color in zip(
        axes,[accs,f1s,cvs],
        ["Test Accuracy (%)", "Macro F1-Score (%)", "CV Mean Accuracy (%)"],
        ["#6ee7b7", "#93c5fd", "#fcd34d"],
    ):
        ax.set_facecolor("#13161e")
        bars = ax.bar(names, vals, color=color, alpha=0.85, width=0.5, edgecolor="none")
        ax.set_title(title, color="white", fontsize=11, pad=10)
        ax.set_ylim(0, 100)
        ax.tick_params(colors="white", labelsize=9)
        ax.set_xticklabels(names, rotation=15, ha="right", color="#9ca3af")
        ax.yaxis.label.set_color("white")
        ax.spines[:].set_color("#333")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,bar.get_height() + 1,f"{val:.1f}%",ha="center", va="bottom", color="white",fontsize=10, fontweight="bold")
        best_idx = names.index(best_name)
        bars[best_idx].set_edgecolor("#fff")
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(ACC_PNG, dpi=120, bbox_inches="tight", facecolor="#0d0f14")
    plt.close()

    cm = np.array(results[best_name]["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d0f14")
    ax.set_facecolor("#13161e")
    sns.heatmap(
        cm_norm, annot=True, fmt=".1f", cmap="YlGn",xticklabels=MOOD_LABELS, yticklabels=MOOD_LABELS,
        linewidths=0.5, linecolor="#1a1e2a",ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Confusion Matrix — {best_name}  (% of actual class)",color="white", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Mood", color="#9ca3af", fontsize=10)
    ax.set_ylabel("Actual Mood",color="#9ca3af", fontsize=10)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=120, bbox_inches="tight", facecolor="#0d0f14")
    plt.close()


def save_everything(best_model, best_name, results):
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(best_model, f)

    report = {
        "trained_at" : datetime.now().isoformat(),
        "dataset" : "tweet_emotions.csv",
        "total_samples" : 40000,
        "best_model" : best_name,
        "mood_labels" : MOOD_LABELS,
        "label_map" : LABEL_MAP,
        "model_results" : results,
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

def main():
    df = load_data()
    df = eda(df)
    df = preprocess(df)

    X = df["clean"].values
    y = df["mood"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results, best_name, best_model = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )
    save_plots(results, best_name, df)
    save_everything(best_model, best_name, results)
    print("Training complete.")

if __name__ == "__main__":
    main()

