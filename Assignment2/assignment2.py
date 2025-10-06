import os
import io
import json
import requests
import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain

from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np

logger = logging.getLogger(__name__)

class CourseRecommendationEngine:
    """Course selection and recommendation engine."""
    def __init__(self, use_azure: bool = False, azure_config: Optional[Dict] = None):
        self.courses_df: Optional[pd.DataFrame] = None
        self.vector_store: Optional[FAISS] = None
        self.retriever = None
        self.use_azure = use_azure
        self.azure_config = azure_config or {}
        self.embeddings = None
        self.llm_chain: Optional[LLMChain] = None

        if self.use_azure:
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=self.azure_config["azure_endpoint"],
                api_key=self.azure_config["azure_api_key"],
                api_version=self.azure_config["azure_api_version"],
                azure_deployment=self.azure_config["embedding_deployment"],
                model="text-embedding-ada-002",
            )
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.azure_config["azure_endpoint"],
                api_key=self.azure_config["azure_api_key"],
                api_version=self.azure_config["azure_api_version"],
                azure_deployment=self.azure_config["chat_deployment"],
                model="gpt-4",
                temperature=0.1,
            )
            prompt = ChatPromptTemplate.from_template(
                "You are an AI learning advisor. User profile: {profile}\n\n"
                "Course Title: {title}\nCourse Description: {description}\n\n"
                "In 1-2 concise sentences, explain why this course is a good next step for the user."
            )
            self.llm_chain = LLMChain(llm=self.llm, prompt=prompt)

    def load_data(self, csv_path_or_url: str) -> bool:
        """Load course dataset from CSV file or HTTP(S) URL."""
        try:
            if str(csv_path_or_url).startswith("http"):
                resp = requests.get(csv_path_or_url, timeout=20)
                resp.raise_for_status()
                self.courses_df = pd.read_csv(io.StringIO(resp.text))
            else:
                self.courses_df = pd.read_csv(csv_path_or_url)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False

        self.courses_df = self.courses_df.rename(columns=str.lower)
        self.courses_df = self.courses_df.rename(
            columns={"course_description": "description", "id": "course_id"}
        )
        for col in ["course_id", "title", "description"]:
            if col not in self.courses_df.columns:
                logger.error("Required columns missing in dataset.")
                return False
        self.courses_df["description"] = self.courses_df["description"].fillna("").astype(str)
        self.courses_df["title"] = self.courses_df["title"].fillna("").astype(str)
        self.courses_df["course_id"] = self.courses_df["course_id"].astype(str)
        return True

    def create_vector_store(self):
        """Build vector store (FAISS) for similarity matching."""
        if self.courses_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        docs = [
            Document(
                page_content=f"Title: {row['title']}\n\nDescription: {row['description']}",
                metadata={"course_id": row["course_id"], "title": row["title"], "description": row["description"]},
            )
            for _, row in self.courses_df.iterrows()
        ]
        if self.use_azure and self.embeddings is not None:
            self.vector_store = FAISS.from_documents(documents=docs, embedding=self.embeddings)
        else:
            tf_texts = [d.page_content for d in docs]
            vect = TfidfVectorizer(max_features=384)
            X = vect.fit_transform(tf_texts).toarray().astype("float32")
            faiss.normalize_L2(X)
            index = faiss.IndexFlatIP(X.shape[1])
            index.add(X)
            self.vector_store = FAISS(embedding_function=None, index=index, docstore=None, index_to_docstore_id=None)
            self._fallback_docs = docs
            self._fallback_vectors = X
        try:
            self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        except Exception:
            self.retriever = None

    def recommend_courses(self, profile: str, completed_ids: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Recommend top_k courses most relevant to the profile."""
        if self.vector_store is None:
            raise ValueError("Vector store not created. Call create_vector_store() first.")
        try:
            results = self.vector_store.similarity_search_with_score(profile, k=min(len(self.courses_df), top_k * 4))
            recs = []
            for doc, score in results:
                cid = str(doc.metadata.get("course_id"))
                if cid in completed_ids:
                    continue
                sim = 1.0 / (1.0 + float(score))
                recs.append((cid, sim))
                if len(recs) >= top_k:
                    break
            return recs[:top_k]
        except Exception:
            if hasattr(self, "_fallback_docs"):
                profile_l = profile.lower()
                scores = []
                profile_words = [w for w in profile_l.split() if len(w) > 3]
                for i, doc in enumerate(self._fallback_docs):
                    text = doc.page_content.lower()
                    matches = sum(1 for w in profile_words if w in text)
                    score = matches / len(profile_words) if profile_words else 0.0
                    scores.append((i, score))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                recs = []
                for idx, score in scores:
                    cid = self._fallback_docs[idx].metadata["course_id"]
                    if cid in completed_ids:
                        continue
                    recs.append((cid, float(score)))
                    if len(recs) >= top_k:
                        break
                return recs
            raise

    def get_course_details(self, course_id: str) -> Optional[Dict]:
        """Look up course by ID."""
        if self.courses_df is None:
            return None
        row = self.courses_df[self.courses_df["course_id"] == str(course_id)]
        if row.empty:
            return None
        r = row.iloc[0]
        return {"course_id": str(r["course_id"]), "title": r["title"], "description": r["description"]}

    def explain_recommendation(self, profile: str, course_id: str) -> str:
        """Generate one-sentence course recommendation explanation."""
        details = self.get_course_details(course_id)
        if not details:
            return ""
        if self.use_azure and self.llm_chain is not None:
            try:
                out = self.llm_chain.run({"profile": profile, "title": details["title"], "description": details["description"]})
                return out.strip()
            except Exception:
                pass
        return f"This course '{details['title']}' is relevant to your profile."


def run_recommendation_batch(
    engine: CourseRecommendationEngine,
    test_cases: List[Tuple[str, str, List[str]]],
    output_path: str
) -> None:
    """Runs a batch of course recommendation scenarios and writes JSON output."""
    output = {"test_cases": []}
    for name, profile, completed in test_cases:
        recs = engine.recommend_courses(profile, completed, top_k=5)
        case_result = {
            "name": name,
            "profile": profile,
            "completed": completed,
            "recommendations": []
        }
        for cid, score in recs:
            details = engine.get_course_details(cid)
            explanation = engine.explain_recommendation(profile, cid)
            case_result["recommendations"].append({
                "course_id": cid,
                "title": details["title"] if details else "",
                "description": details["description"] if details else "",
                "similarity_score": score,
                "explanation": explanation,
            })
        output["test_cases"].append(case_result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Results saved to {output_path}")

# ---- Script/Entry point ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Course Recommendation Batch")
    parser.add_argument("--dataset", type=str, required=True, help="CSV file or URL with course data")
    parser.add_argument("--output", type=str, default="recommendations_output.json", help="Where to save results")
    args = parser.parse_args()

    # Example, in practice get config from env or config file
    AZURE_CONFIG = {
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "embedding_deployment": os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        "chat_deployment": os.getenv("AZURE_CHAT_DELOYMENT"),
    }

    USE_AZURE = False  # Set to True if using Azure

    engine = CourseRecommendationEngine(use_azure=USE_AZURE, azure_config=AZURE_CONFIG if USE_AZURE else None)
    if not engine.load_data(args.dataset):
        logger.error("Failed to load dataset.")
        exit(1)
    engine.create_vector_store()

    test_cases = [
        ("Data Visualization Enthusiast", "I've completed the 'Python Programming for Data Science' course and enjoy data visualization. What should I take next?", ["C016"]),
        ("Azure DevOps Learner", "I know Azure basics and want to manage containers and build CI/CD pipelines. Recommend courses.", ["C007"]),
        ("ML to Deep Learning", "My background is in ML fundamentals; I'd like to specialize in neural networks and production workflows.", ["C001"]),
        ("Microservices & Kubernetes", "I want to learn to build and deploy microservices with Kubernetes—what courses fit best?", []),
        ("Blockchain Beginner", "I'm interested in blockchain and smart contracts but have no prior experience. Which courses do you suggest?", []),
    ]
    run_recommendation_batch(engine, test_cases, args.output)

