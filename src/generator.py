import os
import time
import base64
from pathlib import Path
from openai import OpenAI


class Generator:
    def __init__(self, model="ecnu-plus"):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("ECNU_API_KEY"),
            base_url=os.getenv("ECNU_BASE_URL"),
        )

    # Text RAG
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

    @staticmethod
    def image_to_data_url(image_path):
        path = Path(image_path)

        if path.suffix.lower() in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif path.suffix.lower() == ".png":
            mime = "image/png"
        else:
            raise ValueError(
                f"Unsupported image format: {path.suffix}"
            )

        with open(path, "rb") as f:
            encoded = base64.b64encode(
                f.read()
            ).decode("utf-8")

        return f"data:{mime};base64,{encoded}"

    # Visual RAG
    def generate_visual(self, question, contexts, retries=3):
        prompt = f"""
Answer the question using only the retrieved context.

Choose the answer most directly supported by the context.
Preserve symbols and units exactly.
Return only the short answer with no explanation.

Question:
{question}

Context:
""".strip()

        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for item in contexts:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self.image_to_data_url(
                        item["chunk_path"]
                    )
                },
            })

        content.append({
            "type": "text",
            "text": "Answer:",
        })

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    temperature=0,
                )

                return (
                    response.choices[0]
                    .message.content.strip()
                )

            except Exception as e:
                if attempt == retries - 1:
                    raise e

                time.sleep(2 ** attempt)