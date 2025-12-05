import pandas as pd
import glob
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RAGSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.data = []
        self.vectors = None

    def load_files(self):
        csv_files = glob.glob("*.csv")
        for file in csv_files:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                content = f"Product: {row.get('name', '')}. Details: {row.get('target_concerns', '')} {row.get('contraindications_short', '')}"
                self.data.append({"text": content, "source": file})

        md_files = glob.glob("*.md")
        for file in md_files:
            with open(file, 'r', encoding='utf-8') as f:
                sections = f.read().split("## ")
                for s in sections:
                    if s.strip():
                        self.data.append({"text": "## " + s.strip(), "source": file})
        
        print(f"Loaded {len(self.data)} items.")

    def build_index(self):
        texts = [d['text'] for d in self.data]
        self.vectors = self.model.encode(texts)

    def search(self, query):
        q_vec = self.model.encode([query])
        scores = cosine_similarity(q_vec, self.vectors)[0]
        top_idx = scores.argsort()[-3:][::-1]
        
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        for i in top_idx:
            item = self.data[i]
            print(f"Source: {item['source']}")
            print(f"Content: {item['text'][:200]}...\n")

if __name__ == "__main__":
    rag = RAGSystem()
    rag.load_files()
    rag.build_index()
    
    while True:
        user_query = input("\nEnter your question (or type 'exit' to quit): ")
        if user_query.lower() in ['exit', 'quit']:
            break
        rag.search(user_query)