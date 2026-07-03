import os

import gradio as gr

from apex_rag import ApexIndex
from apex_rag.providers import GroqProvider, OllamaProvider, OpenAIProvider

# Configuration
DB_URL = "sqlite+aiosqlite:///hgf_space.db"

# Global state to hold the index instance
index_instance = None


async def get_index():
    global index_instance
    if index_instance is None:
        # Priority: Groq -> OpenAI -> Ollama
        if os.getenv("GROQ_API_KEY"):
            provider = GroqProvider()
        elif os.getenv("OPENAI_API_KEY"):
            provider = OpenAIProvider()
        else:
            # Fallback to local Ollama (might not work in HGF Space without setup)
            provider = OllamaProvider(model="phi3")

        index_instance = await ApexIndex.create(db_url=DB_URL, model=provider)
    return index_instance


async def process_file(file):
    if file is None:
        return "Please upload a file."

    index = await get_index()
    doc_id = await index.ingest(file.name)
    return doc_id


async def ask_question(doc_id, question):
    if not doc_id or not question:
        return "Please ingest a document and enter a question."

    index = await get_index()

    # Use the orchestrator for high-precision retrieval
    answer = await index.orchestrate_query(question, doc_id)

    if not answer:
        return "I couldn't find a verified answer in the document."

    return answer


# Define the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏔️ ApexRAG Demo")
    gr.Markdown(
        "### Structural AI Retrieval — Reading documents like a human, not a bag of chunks."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF or Markdown")
            ingest_btn = gr.Button("Ingest Document", variant="primary")
            doc_id_state = gr.State()
            ingest_status = gr.Markdown("Status: Ready")

        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask a question about the document",
                placeholder="e.g., Compare the revenue growth between Q2 and Q3.",
            )
            ask_btn = gr.Button("Get Answer", variant="primary")
            answer_output = gr.Textbox(label="ApexRAG Answer", interactive=False, lines=10)

    # Event Handlers
    async def handle_ingest(file):
        doc_id = await process_file(file)
        return doc_id, f"Status: Ingested (ID: {doc_id})"

    ingest_btn.click(handle_ingest, inputs=[file_input], outputs=[doc_id_state, ingest_status])

    ask_btn.click(ask_question, inputs=[doc_id_state, question_input], outputs=[answer_output])

if __name__ == "__main__":
    demo.launch()
