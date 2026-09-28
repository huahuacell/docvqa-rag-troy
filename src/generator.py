import os
import time
from openai import OpenAI


class Generator:
    def __init__(self, model="ecnu-plus"):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("ECNU_API_KEY"),
            base_url=os.getenv("ECNU_BASE_URL"),
        )

    def generate_text(self, question, contexts, retries=3):
        context_text = "\n\n".join(
            f"[Context {i+1}]\n{x['text']}"
            for i, x in enumerate(contexts)
        )

        prompt = f"""
Answer the question using only the retrieved context.

Choose the answer most directly supported by the context.
Preserve symbols and units exactly.
Return only the short answer with no explanation.

Question:
{question}

Context:
{context_text}

Answer:
""".strip()

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                time.sleep(2 ** attempt)