
import re

from sklearn.metrics.pairwise import cosine_similarity


SKILLS = {
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript"],
    "typescript": ["typescript"],
    "c++": ["c++"],
    "sql": ["sql", "mysql", "postgresql", "sqlite"],
    "mongodb": ["mongodb"],
    "react": ["react", "react.js", "reactjs"],
    "node.js": ["node.js", "nodejs"],
    "html": ["html"],
    "css": ["css"],
    "flask": ["flask"],
    "django": ["django"],
    "fastapi": ["fastapi"],
    "machine learning": [
        "machine learning",
        "scikit-learn",
        "sklearn",
    ],
    "deep learning": [
        "deep learning",
        "pytorch",
        "tensorflow",
    ],
    "data analysis": ["data analysis", "data analytics"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "aws": ["aws", "amazon web services"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes"],
    "git": ["git", "github"],
    "rest api": ["rest api", "restful api"],
    "linux": ["linux"],
    "cybersecurity": [
        "cybersecurity",
        "information security",
    ],
    "communication": ["communication skills"],
    "problem solving": [
        "problem solving",
        "problem-solving",
    ],
}


def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).lower(),
    ).strip()


def extract_skills(text):
    text = normalize_text(text)
    found = []

    for skill, aliases in SKILLS.items():
        for alias in aliases:
            pattern = (
                r"(?<!\w)"
                + re.escape(alias.lower())
                + r"(?!\w)"
            )

            if re.search(pattern, text):
                found.append(skill)
                break

    return sorted(found)


def analyze_match(resume_text, job_text, vectorizer):
    if not resume_text.strip() or not job_text.strip():
        raise ValueError(
            "Resume and job description cannot be empty."
        )

    vectors = vectorizer.transform(
        [resume_text, job_text]
    )

    similarity = (
        cosine_similarity(
            vectors[0],
            vectors[1],
        )[0, 0] * 100
    )

    resume_skills = set(
        extract_skills(resume_text)
    )

    job_skills = set(
        extract_skills(job_text)
    )

    matched = sorted(
        resume_skills & job_skills
    )

    missing = sorted(
        job_skills - resume_skills
    )

    coverage = (
        len(matched) / len(job_skills) * 100
        if job_skills
        else None
    )

    return {
        "similarity": round(float(similarity), 2),
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_coverage": (
            round(coverage, 2)
            if coverage is not None
            else None
        ),
    }


def find_similar_resumes(
    resume_text,
    vectorizer,
    knn_model,
    reference_df,
    k=5,
):
    query_vector = vectorizer.transform(
        [resume_text]
    )

    k = min(k, len(reference_df))

    distances, indices = knn_model.kneighbors(
        query_vector,
        n_neighbors=k,
    )

    results = []

    for distance, index in zip(
        distances[0],
        indices[0],
    ):
        row = reference_df.iloc[index]

        results.append({
            "category": row["category"],
            "similarity": round(
                max(
                    0.0,
                    1.0 - float(distance),
                ) * 100,
                2,
            ),
            "resume_preview": str(
                row["resume"]
            )[:300],
        })

    return results
