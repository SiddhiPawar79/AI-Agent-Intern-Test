from pathlib import Path
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


KB_PATH = Path("knowledge-base")


def load_documents():
    files = list(KB_PATH.glob("*.md"))

    documents = []

    for file in files:
        text = file.read_text(encoding="utf-8")

        # Supplied files contain metadata followed by ---
        parts = text.split("---", 1)

        metadata_text = parts[0]
        content = parts[1] if len(parts) > 1 else text

        metadata = {}

        for line in metadata_text.splitlines():
            line = line.strip()

            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

        documents.append({
            "filename": file.name,
            "metadata": metadata,
            "content": content.strip()
        })

    return documents


def create_chunks(documents):
    chunks = []

    for doc in documents:

        sections = re.split(
            r"\n(?=#{1,3}\s)",
            doc["content"]
        )

        for section in sections:

            section = section.strip()

            if not section:
                continue

            heading_match = re.match(
                r"^(#{1,3})\s+(.+)",
                section
            )

            if heading_match:
                heading = heading_match.group(2).strip()
            else:
                heading = "General"

            chunks.append({
                "text": section,
                "filename": doc["filename"],
                "heading": heading,
                "metadata": doc["metadata"]
            })

    return chunks


class RAGRetriever:

    def __init__(self):

        self.documents = load_documents()

        self.chunks = create_chunks(
            self.documents
        )

        if not self.chunks:
            raise ValueError(
                "No knowledgebase chunks were created."
            )

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embedding_matrix = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = int(
            embedding_matrix.shape[1]
        )

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embedding_matrix
        )

    def retrieve(self, query, top_k=5):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            chunk = self.chunks[idx]

            results.append({
                "score": float(score),
                "filename": chunk["filename"],
                "heading": chunk["heading"],
                "metadata": chunk["metadata"],
                "text": chunk["text"]
            })

        return results


if __name__ == "__main__":

    rag = RAGRetriever()

    print(
        "Documents:",
        len(rag.documents)
    )

    print(
        "Chunks:",
        len(rag.chunks)
    )

    results = rag.retrieve(
        "How long does a regular customer "
        "have to return an unused backpack?"
    )

    for result in results:

        print("\n---")
        print("Score:", result["score"])
        print("Source:", result["filename"])
        print("Heading:", result["heading"])
        print("Text:", result["text"][:500])
