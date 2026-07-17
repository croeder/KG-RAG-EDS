#!/usr/bin/env python3
"""End-to-end plain vector RAG query (README build seq #1, generation step).

question -> embed -> retrieve top-k node texts from DuckDB -> build a grounded
prompt -> local LLM writes the answer.

The generation backend sits behind one seam: build_prompt() produces
backend-neutral (system, user) text; generate() wraps and sends it. Swapping the
local model for the Anthropic API later means replacing only generate()'s body —
no re-embedding, because the handoff to the generator is text, not vectors.
"""

import sys

import duckdb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_HOME = "/Users/croeder/git/KG-RAG-EDS"
DB = f"{PROJECT_HOME}/data/eds.duckdb"

EMBED_MODEL = "all-MiniLM-L6-v2"      # must match what embed_nodes.py used
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DIM = 384
K = 5                                  # how many node texts to retrieve

SYSTEM = (
    "You are a biomedical assistant. Answer the question using ONLY the context "
    "passages provided, which describe Ehlers-Danlos syndrome and related "
    "entities. If the answer is not in the context, say you do not have enough "
    "information rather than guessing. Be concise."
)


def retrieve(con, embedder, question, k=K):
    """Return the k nearest node rows as (id, text, similarity)."""
    qv = embedder.encode(question).tolist()
    return con.execute(
        """
        SELECT id, text, array_cosine_similarity(embedding, ?::FLOAT[384]) AS sim
        FROM nodes
        ORDER BY sim DESC
        LIMIT ?
        """,
        [qv, k],
    ).fetchall()


def build_prompt(question, hits):
    """Assemble backend-neutral (system, user) text from retrieved node texts."""
    context = "\n".join(f"[{id_}] {text}" for id_, text, _ in hits)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return SYSTEM, user


def generate(system, user):
    """Local Qwen backend. To use the Anthropic API instead, replace this body
    with a client.messages.create(...) call — nothing else changes."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device)

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(device)
    generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    # slice off the prompt tokens; decode only the newly generated answer
    new_tokens = generated[0][inputs.input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    question = " ".join(sys.argv[1:]) or "What joint problems are associated with EDS?"

    con = duckdb.connect(DB, read_only=True)
    embedder = SentenceTransformer(EMBED_MODEL)

    hits = retrieve(con, embedder, question)
    system, user = build_prompt(question, hits)
    answer = generate(system, user)

    print(f"Q: {question}\n")
    print("Retrieved (grounding):")
    for id_, text, sim in hits:
        print(f"  {sim:.3f}  {id_:14} {text[:60]}")
    print(f"\nA: {answer}")


if __name__ == "__main__":
    main()
