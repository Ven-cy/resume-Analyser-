
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Resume" / "Resume_Computer_Engineering_Updated.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Existing categories + additional Computer Engineering roles
CATEGORIES = [
    # Existing categories
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL",
    "ARTS", "AUTOMOBILE", "AVIATION", "BANKING", "BPO",
    "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION",
    "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA",
    "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE",
    "HR", "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS",
    "SALES", "TEACHER",

    # Software development
    "SOFTWARE-ENGINEER",
    "SOFTWARE-DEVELOPER",
    "FULL-STACK-DEVELOPER",
    "FRONTEND-DEVELOPER",
    "BACKEND-DEVELOPER",
    "WEB-DEVELOPER",
    "MOBILE-APP-DEVELOPER",
    "ANDROID-DEVELOPER",
    "IOS-DEVELOPER",
    "PYTHON-DEVELOPER",
    "JAVA-DEVELOPER",
    "DOTNET-DEVELOPER",
    "PHP-DEVELOPER",
    "REACT-DEVELOPER",
    "NODEJS-DEVELOPER",

    # Data science and AI
    "DATA-ANALYST",
    "DATA-ENGINEER",
    "DATA-SCIENTIST",
    "MACHINE-LEARNING-ENGINEER",
    "AI-ENGINEER",
    "DEEP-LEARNING-ENGINEER",
    "NLP-ENGINEER",
    "COMPUTER-VISION-ENGINEER",
    "BUSINESS-INTELLIGENCE-ANALYST",

    # Cloud, DevOps and infrastructure
    "CLOUD-ENGINEER",
    "AWS-ENGINEER",
    "AZURE-ENGINEER",
    "DEVOPS-ENGINEER",
    "SITE-RELIABILITY-ENGINEER",
    "PLATFORM-ENGINEER",
    "SYSTEMS-ADMINISTRATOR",
    "NETWORK-ENGINEER",
    "DATABASE-ADMINISTRATOR",
    "DATABASE-DEVELOPER",

    # Cybersecurity
    "CYBERSECURITY-ANALYST",
    "SECURITY-ENGINEER",
    "PENETRATION-TESTER",
    "ETHICAL-HACKER",
    "SOC-ANALYST",
    "DIGITAL-FORENSICS-ANALYST",

    # Software testing
    "QA-ENGINEER",
    "SOFTWARE-TESTER",
    "AUTOMATION-TEST-ENGINEER",
    "PERFORMANCE-TEST-ENGINEER",

    # Computer engineering and emerging technologies
    "EMBEDDED-SYSTEMS-ENGINEER",
    "IOT-ENGINEER",
    "ROBOTICS-ENGINEER",
    "FIRMWARE-ENGINEER",
    "COMPUTER-HARDWARE-ENGINEER",
    "VLSI-ENGINEER",
    "BLOCKCHAIN-DEVELOPER",
    "GAME-DEVELOPER",
    "UI-UX-DESIGNER",
    "TECHNICAL-SUPPORT-ENGINEER",
    "IT-SUPPORT-ENGINEER",
    "SYSTEMS-ENGINEER",
    "SOFTWARE-ARCHITECT",
    "TECHNICAL-PROJECT-MANAGER",
    "IT-BUSINESS-ANALYST",
    "ERP-DEVELOPER",
]


def find_column(df, candidates):
    columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]

    return None


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}"
        )

    print("Loading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    print("CSV columns:", df.columns.tolist())

    resume_col = find_column(
        df,
        [
            "resume_str",
            "resume",
            "resume_text",
            "resume_html",
            "text",
            "content",
        ],
    )

    category_col = find_column(
        df,
        ["category", "job_category", "label", "class"],
    )

    if resume_col is None or category_col is None:
        raise ValueError(
            "Could not identify resume and category columns."
        )

    df = df[[resume_col, category_col]].copy()
    df.columns = ["resume", "category"]

    df["resume"] = (
        df["resume"].fillna("").astype(str).str.strip()
    )
    df["category"] = (
        df["category"].fillna("").astype(str).str.strip()
    )

    df = df[
        (df["resume"] != "") &
        (df["category"] != "")
    ]

    df = df.drop_duplicates(
        subset=["resume", "category"]
    ).reset_index(drop=True)

    # Only categories with actual training data can be learned.
    actual_categories = sorted(df["category"].unique())

    print("\nRequested categories:", len(CATEGORIES))
    print("Categories present in CSV:", len(actual_categories))

    print("\nCategories in dataset:")
    for category in actual_categories:
        print("-", category)

    # Report categories for which training examples are missing.
    missing_categories = sorted(
        set(CATEGORIES) - set(actual_categories)
    )

    if missing_categories:
        print(
            "\nThese added categories have no matching training "
            "examples yet:"
        )
        for category in missing_categories:
            print("-", category)

    counts = df["category"].value_counts()
    rare = counts[counts < 2]

    if not rare.empty:
        raise ValueError(
            "At least two records are required per category "
            "for stratified train/test splitting.\n"
            f"Rare categories: {rare.to_dict()}"
        )

    if df["category"].nunique() < 2:
        raise ValueError("At least two categories are required.")

    X_train_text, X_test_text, y_train, y_test = (
        train_test_split(
            df["resume"],
            df["category"],
            test_size=0.2,
            random_state=42,
            stratify=df["category"],
        )
    )

    print("\nConverting resume text into TF-IDF features...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype="float32",
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
        "K-NN": KNeighborsClassifier(
            n_neighbors=5,
            metric="cosine",
            algorithm="brute",
            weights="distance",
        ),
    }

    metrics = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"Test accuracy: {accuracy * 100:.2f}%")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        filename = (
            "logistic_regression.pkl"
            if name == "Logistic Regression"
            else "knn.pkl"
        )

        joblib.dump(model, MODEL_DIR / filename)

        metrics[name] = {
            "accuracy": round(float(accuracy), 4)
        }

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.pkl",
    )

    with open(
        MODEL_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=4)

    with open(
        MODEL_DIR / "categories.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(actual_categories, f, indent=4)

    reference_df = pd.DataFrame({
        "resume": X_train_text.reset_index(drop=True),
        "category": y_train.reset_index(drop=True),
    })

    reference_df.to_pickle(
        MODEL_DIR / "resume_reference.pkl"
    )

    print("\nTraining completed successfully.")
    print(
        "Models saved in:",
        MODEL_DIR,
    )
    print(
        "Remember: categories without labeled resumes "
        "cannot be predicted reliably."
    )


if __name__ == "__main__":
    main()

