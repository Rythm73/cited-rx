import os
os.environ["GRADIO_TEMP_DIR"] = "/tmp"

import gradio as gr
from pathlib import Path

from backend.pipeline import run_pipeline
from backend.ingest import ingest_pdf
from backend.state import app_state
from config import DEFAULT_CORPUS

DEFAULT_CORPUS_LABEL = "AHA/ACC CCD performance measures (bundled)"

def query_rag(message: str, active_corpus_id: str, active_corpus_label: str) -> str:
    if not message or not message.strip():
        return "Please enter a question."
    try:
        client = app_state.get("qdrant")
        if not client:
            return "Database connection is not ready yet."
        result = run_pipeline(message, qdrant_client=client, top_k=5, corpus_id=active_corpus_id)
        confidence_pct = int(result.confidence * 100)
        if result.refused:
            output = f"⚠️ **Refused** *(out of corpus — confidence: {confidence_pct}%)*\n\n"
        else:
            output = f"**Confidence: {confidence_pct}%** *(querying: {active_corpus_label})*\n\n"
        output += result.rendered_answer + "\n"
        return output
        
    except Exception as e:
        return f"Unexpected error: {type(e).__name__}: {e}"

def user_then_assistant(user_msg, history, corpus_id_val, corpus_label_val):
    if history is None:
        history = []
    if not user_msg or not user_msg.strip():
        return "", history
    response = query_rag(user_msg, corpus_id_val, corpus_label_val)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})
    return "", history


def handle_upload(file_obj, chat_history):
    if chat_history is None:
        chat_history = []
    if file_obj is None:
        yield (DEFAULT_CORPUS, DEFAULT_CORPUS_LABEL, f"**Active corpus:** {DEFAULT_CORPUS_LABEL}", chat_history)
        return
    pdf_path = file_obj.name
    pdf_name = Path(pdf_path).name
    chat_history.append({"role": "assistant", "content": f"⏳ **Indexing `{pdf_name}`...** Typically 5-10 minutes for a 50-page PDF. Don't close this tab."})
    yield (DEFAULT_CORPUS, DEFAULT_CORPUS_LABEL, f"⏳ **Indexing `{pdf_name}`...**", chat_history)
    try:
        client = app_state.get("qdrant")
        result = ingest_pdf(pdf_path, source_doc=pdf_name, qdrant_client=client)
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"❌ Upload failed: {type(e).__name__}: {e}"})
        yield (DEFAULT_CORPUS, DEFAULT_CORPUS_LABEL, f"**Active corpus:** {DEFAULT_CORPUS_LABEL}", chat_history)
        return
    new_corpus_id = result["corpus_id"]
    n_chunks = result["n_chunks"]
    n_pages = result["n_pages"]
    chat_history.append({"role": "assistant", "content": f"✅ Indexed `{pdf_name}` — {n_chunks} chunks across {n_pages} pages. Subsequent questions will query this document."})
    yield new_corpus_id, pdf_name, f"**Active corpus:** {pdf_name}", chat_history


def reset_to_default(chat_history):
    if chat_history is None:
        chat_history = []
    chat_history.append({"role": "assistant", "content": f"↩ Switched back to default: {DEFAULT_CORPUS_LABEL}."})
    return (DEFAULT_CORPUS, DEFAULT_CORPUS_LABEL, f"**Active corpus:** {DEFAULT_CORPUS_LABEL}", chat_history, None)


with gr.Blocks(title="cited-rx") as demo:
    active_corpus_id = gr.State(DEFAULT_CORPUS)
    active_corpus_label = gr.State(DEFAULT_CORPUS_LABEL)

    gr.Markdown("# cited-rx — Grounded RAG with Page-Level Citations")
    gr.Markdown(
        "Ask questions about the bundled cardiology corpus (2025 AHA/ACC Clinical "
        "Performance and Quality Measures for Chronic Coronary Disease), or upload "
        "your own PDF. Answers are grounded in the source document with page-level "
        "citations and a confidence gate that refuses out-of-corpus questions."
    )

    with gr.Row():
        with gr.Column(scale=3):
            upload = gr.File(
                label="Upload a PDF (optional — 5-10 min indexing on CPU)",
                file_types=[".pdf"],
                file_count="single",
            )
        with gr.Column(scale=1):
            reset_btn = gr.Button("↩ Reset to bundled corpus", variant="secondary")

    corpus_status = gr.Markdown(f"**Active corpus:** {DEFAULT_CORPUS_LABEL}")

    chatbot = gr.Chatbot(label="Conversation", height=300)
    msg = gr.Textbox(
        label="Your question",
        placeholder="Ask about the active corpus...",
        lines=3,
        min_width=400,
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear chat")

    gr.Examples(
        examples=[
            "What is the recommended LDL cholesterol target?",
            "What are performance measures for cardiovascular care?",
            "How are quality measures for chronic coronary disease developed?",
            "What is the role of medication adherence in CCD performance measures?",
            "What is the capital of France?",
        ],
        inputs=msg,
    )

    upload.upload(handle_upload, inputs=[upload, chatbot], outputs=[active_corpus_id, active_corpus_label, corpus_status, chatbot])
    reset_btn.click(reset_to_default, inputs=[chatbot], outputs=[active_corpus_id, active_corpus_label, corpus_status, chatbot, upload])
    send_btn.click(user_then_assistant, inputs=[msg, chatbot, active_corpus_id, active_corpus_label], outputs=[msg, chatbot])
    msg.submit(user_then_assistant, inputs=[msg, chatbot, active_corpus_id, active_corpus_label], outputs=[msg, chatbot])
    clear_btn.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    demo.launch()