#!/usr/bin/env python3
"""End-to-end plain vector RAG query (README build seq #1, generation step).

question -> embed -> retrieve top-k node texts from DuckDB -> build a grounded
prompt -> an LLM writes the answer.

The generation backend sits behind one seam: build_prompt() produces
backend-neutral (system, user) text; generate() wraps and sends it. The local
Qwen model and the Anthropic API are two bodies behind that same seam, chosen by
the RAG_BACKEND env var (default "local", set "anthropic" for the API). Switching
costs nothing on the retrieval side — the handoff to the generator is text, not
vectors, so no re-embedding is involved.
"""

import os
import sys

import duckdb
from sentence_transformers import SentenceTransformer

PROJECT_HOME = "/Users/croeder/git/KG-RAG-EDS"
DB = f"{PROJECT_HOME}/data/eds.duckdb"

EMBED_MODEL = "all-MiniLM-L6-v2"  # must match what embed_nodes.py used
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ANTHROPIC_MODEL = "claude-opus-4-8"
DIM = 384
K = 5  # top-k: how many of the nearest node texts to keep as grounding

BACKEND = os.environ.get("RAG_BACKEND", "local")  # "local" | "anthropic"

SYSTEM = (
    "You are a biomedical assistant. Answer the question using ONLY the context "
    "passages provided, which describe Ehlers-Danlos syndrome and related "
    "entities. If the answer is not in the context, say you do not have enough "
    "information rather than guessing. Be concise."
)


def retrieve(con, embedder, question, k=K):
    """Return the k nearest node rows as (id, text, similarity).

    k is the "top-k": we sort every node by similarity to the question and keep
    only the k closest. Those k node texts become the grounding handed to the LLM.
    """
    question_vector = embedder.encode(question).tolist()

    sql = """
        SELECT id,
               text,
               array_cosine_similarity(embedding, ?::FLOAT[384]) AS sim
        FROM nodes
        ORDER BY sim DESC     -- most similar first
        LIMIT ?               -- keep only the top k
    """
    params = [question_vector, k]        # fills the two ? above, in order

    rows = con.execute(sql, params).fetchall()
    return rows


def build_prompt(question, hits):
    """Assemble backend-neutral (system, user) text from retrieved node texts."""
    context = "\n".join(f"[{id_}] {text}" for id_, text, _ in hits)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return SYSTEM, user


def generate(system, user):
    """The seam: same (system, user) text, dispatched to the chosen backend."""
    if BACKEND == "anthropic":
        return generate_anthropic(system, user)
    return generate_local(system, user)


def generate_local(system, user):
    """Local Qwen via transformers. torch/transformers are imported here, not at
    module top, so the anthropic backend doesn't pay to load them."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device)

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(device)
    generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    # slice off the prompt tokens; decode only the newly generated answer
    new_tokens = generated[0][inputs.input_ids.shape[1] :]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def generate_anthropic(system, user):
    """Anthropic API. Same (system, user) content as the local path; only the
    envelope differs — system goes to the `system` field, the context+question
    to a single user message. Auth resolves from ANTHROPIC_API_KEY."""
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def main():
    question = " ".join(sys.argv[1:]) or "What joint problems are associated with EDS?"

    con = duckdb.connect(DB, read_only=True)
    embedder = SentenceTransformer(EMBED_MODEL)

    hits = retrieve(con, embedder, question)
    system, user = build_prompt(question, hits)
    answer = generate(system, user)

    print(f"Q: {question}   [backend: {BACKEND}]\n")
    print("Retrieved (grounding):")
    for id_, text, sim in hits:
        print(f"  {sim:.3f}  {id_:14} {text[:60]}")
    print(f"\nA: {answer}")


if __name__ == "__main__":
    main()
