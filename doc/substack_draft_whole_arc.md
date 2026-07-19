# Building KG-RAG by Hand (and Letting the Failures Teach Me)


---

I wanted to actually understand KG-RAG, not import a library that does it and call it learning. So I set a rule: build it
by hand, slowly, against a real biomedical knowledge graph, and don't move to the next piece until I understand the one
in front of me. I paired with an LLM the whole way — but as a navigator I could overrule, not an autopilot. More on that
working style at the end, because it turned out to matter as much as the code.

The graph is the Monarch Initiative KG, scoped down to Ehlers-Danlos syndrome (EDS) so it's small enough to work with quickly: about 800 nodes, 1900 edges. Everything below is that slice.

All the code is on GitHub: [github.com/croeder/KG-RAG-EDS](https://github.com/croeder/KG-RAG-EDS). The `bin/` scripts named throughout are there.


## Starting with a baseline on what embedding is, and how transformers are different
A key task in RAG architectures is finding bits of information relevant to a question. A really basic way to do that would 
be to code your question as either a regular expression or an SQL "like" query. Those are surface methods focussed on the
letters. An embedding like what word2vec would create moves into the realm of meaning. The algorithm, a shallow neural net, 
creates an array of numbers for each word. It works out that words with similar usage and meaning get similar arrays,
so you can compute the angle between their two many dimensional vectors. A small angle means they point the same way —
similar meaning. The cosine of that angle is the similarity score itself: it runs from -1 to 1, and a small angle gives a
cosine near 1, so a *higher* number means more similar. Transformer architectures take it a step further where the surrounding text influences how such a vector
is created for a word, so that you can deal with different meanings.

One more step gets us to what RAG actually uses. So far this is all *per word*, but for retrieval I don't want a vector
per word — I want a single vector for a whole node's text, and a single vector for the whole question, so I can compare
the two directly. That's what a *sentence-transformer* does: it runs the passage through a transformer and pools the
per-word vectors into one vector for the entire chunk. So everywhere below, "embed" means turning a whole piece of text —
a node's description, or my question — into one vector, not embedding words one at a time.

## An early misconception about what RAG even is

Here's the first thing I got wrong. I thought RAG *starts*
with a pile of documents, builds a knowledge graph out of them, and then answers questions against that graph. So in my
head, "RAG" and "KG-RAG" were the same thing.

They're not. Plain RAG is just: take a corpus, embed it into vectors, retrieve the chunks nearest your question, hand them to an LLM to
write an answer. The corpus can be anything. It doesn't have to be a graph, and if it *is* a graph, plain RAG treats it as
a bag of text and ignores the structure entirely.

Which is exactly what I did in project 1. I had a knowledge graph, but I used it as a text corpus — I flattened each node
into a sentence and embedded those sentences into vectors. That's plain RAG that happens to be pointed at a KG. Using the graph *as a
graph* — following its edges — is the thing that earns the name KG-RAG, and that was project 2. Realizing my mental model
was backwards (RAG doesn't require a KG; I just happened to start with one) was the first real unlock.

## Four sub-projects:
### Project 1: plain vector RAG, and the embedder/generator split

Scripts: `bin/rag_query.py` (end-to-end), built on `bin/node_text.py` and `bin/embed_nodes.py` (the one-time embed).

The mechanics are simple once you see them. A sentence-transformer (`all-MiniLM-L6-v2`) turns each node's text into a
384-number vector. The *same* model turns my question into a vector. Retrieval is then just "which node vectors point in
the same direction as the question vector" — cosine similarity, which DuckDB does natively. Top-k nearest nodes come back,
you stuff their text versions into a prompt, an LLM writes the answer.

Embedding text into a vector is one call, `model.encode(...)`. The node side runs once, ahead of time — it embeds
every node's text into vectors and stores them:

```python
# embed_nodes.py — embed all ~800 node texts into vectors in one batch, then store them
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)          # texts = every node's description
```

The question side runs at query time — the *same* model embeds the question into a vector, one string instead of a list:

```python
# rag_query.py — embed the question into a vector the same way, so it's comparable
question_vector = embedder.encode(question).tolist()
```

That whole idea is a short bit of SQL — the retrieval *is* the cosine sort. Score every node against the question,
put the most similar first, and keep only the top k:

```sql
-- rag_query.py
SELECT id,
       text,
       array_cosine_similarity(embedding, ?::FLOAT[384]) AS sim
FROM nodes
ORDER BY sim DESC     -- most similar first
LIMIT ?               -- keep only the top k
```

Two things clicked here that I hadn't internalized:

**The embedder and the generator are different models, and they don't share a vector space.** The little
sentence-transformer that does retrieval and the big LLM that writes the answer are separate. The handoff between them
isn't vectors — it's plain text. That sounds obvious written down, but it's why you can swap the generator (I put local
Qwen and the Anthropic API behind a single seam, chosen by an env var) without re-embedding a thing. Retrieval doesn't
care who writes the answer.

The seam is just this — same `(system, user)` text, dispatched by an env var:

```python
# rag_query.py
if BACKEND == "anthropic": return generate_anthropic(system, user)
return generate_local(system, user)
```

**Semantic retrieval is a genuinely different tool than SQL.** "Loose joints that dislocate easily" finds *Joint
subluxation* without sharing a word with it. You cannot do that with `LIKE '%...%'` and a pile of `toupper()` calls. That
fuzziness is the whole point — and, foreshadowing, also the source of a lot of grief later.

### Project 2: using the graph as a graph

Scripts: `bin/project_2_anchor.py`, `project_2_predicate_classifier.py`, `project_2_disambiguate.py`, `project_2_traverse.py`, wired together in `project_2_kg_rag_query.py`.

Starting with a simple introduction to graphs for those that haven't seen them. They are basically
triples: subject, predicate, object. "EDS", "has_phenotype", "loose joints", for rough example. Subject and object
are nodes. Predicates are edges between them.

This is where it got interesting, because retrieving by structure needs a few things plain RAG never did:

1. **Anchor** — a starter node for the graph traversal. You get it from the words of the question. "What genes cause EDS?" has to resolve "EDS" to an
   actual node in the graph before you can do anything.
2. **Predicates** — edges to follow to other nodes in the graph. They also come from the question by way of embedding.
3. **Traverse** — from that node, follow the identified predicates and collect the connected facts.

The clean insight that organized all of project 2: **embeddings only appear at the entry.** You embed the question into a
vector twice — once to find the anchor, once to pick the predicate — and everything after that is pure graph. The two are
separate matches against separate sets of vectors, so I'll take them one at a time.

**Finding the anchor.** This is a cosine match against the node vectors — the same ones project 1 built. Embed the
question into a vector, and take the nearest node(s) as candidates for what the question is about:

```python
# project_2_anchor.py — the nearest nodes to the question are the anchor candidates
question_vector = embedder.encode(question).tolist()
candidates = con.execute("""
    SELECT id, category, text,
           array_cosine_similarity(embedding, ?::FLOAT[384]) AS sim
    FROM nodes
    ORDER BY sim DESC
    LIMIT ?
""", [question_vector, k]).fetchall()
```

**Picking the predicates.** A separate cosine match, against a different set of vectors: not the nodes this time, but a
short hand-written description of each predicate. Embed the question into a vector, score it against every predicate
description, and keep the ones above a cutoff:

```python
# project_2_predicate_classifier.py — rank predicates by similarity to the question
q = model.encode(question, convert_to_tensor=True)
sims = util.cos_sim(q, pred_emb)[0]                       # one score per predicate
ranked = sorted(zip(pred_ids, sims.tolist()), key=lambda r: r[1], reverse=True)
picked = [p for p, score in ranked if score >= cutoff]   # keep those above the cutoff
```

**Traversing.** That is the last of the embeddings. With an anchor node and the picked predicate(s) in hand, the rest is
pure graph — SQL walking edges. The traverse follows those predicate edges out of the anchor and collects the facts one
hop away, and those facts — not any vector — become the grounding the LLM writes the answer from. That's the graph doing
the work, which is the entire reason to bother with KG-RAG.

And then I spent the rest of the project walking into instructive walls.

#### Wall 1: the authoritative definitions were worse than my sloppy ones

To pick which predicate a question is about, I embedded a short description of each predicate into a vector and matched it against the
question. My first instinct was to use the *official* Biolink Model definitions — authoritative, curated, surely better
than anything I'd scribble. For example, Biolink has:
```
  ▎ holds between a biological entity and a phenotype, where a phenotype is construed broadly as any kind of quality of an organism
  ▎ part, a collection of these qualities, or a change in quality or qualities (e.g. abnormally increased temperature). In SNOMEDCT,
  ▎ disorders with keyword 'characterized by' should translate into this predicate.
```

They were worse. A question about symptoms matched *mode of inheritance* instead of *has phenotype*. The
problem: ontology definitions are written in ontology language, which shares almost no words — and therefore almost no
embedding neighborhood — with the way a person actually asks a question. I replaced them with plain, question-facing
descriptions I wrote by hand ("symptoms, clinical features, signs...") and suddenly every test question landed on the
right predicate. Authoritative ≠ useful when you're matching against casual language.

I switched to:
``` 
"biolink:has_phenotype": symptoms, clinical features, signs, or phenotypes of the disease
```

#### Wall 2: my hand-written descriptions were magic numbers

I was pleased with those hand-written descriptions right up until I realized that they were exactly the kind of
buried, hand-tuned magic values I'd complain about in anyone else's code. Guilty. They went into a config file, I added a
linter (ruff), and I wrote a loud comment explaining *why* they're hand-written and not pulled from the ontology — so the
next person (me, in a month) doesn't "fix" it back to the authoritative-but-useless version.

#### Wall 3: the edge direction I assumed didn't exist

I was about to write the traversal as one fixed query — `WHERE subject = anchor` — when I stopped to actually check the
data. Good thing. The disease is *not* always the subject of its edges. It's the subject of `has_phenotype` (disease →
symptom) but the **object** of `causes` (gene → disease; the graph actually names that predicate
`gene_associated_with_condition`, but I'll write `causes` for readability). Guess subject-only and "what genes cause EDS?" returns nothing,
silently. The fix was to match the anchor on *either* side of an edge and take whatever's on the other end — no
per-predicate bookkeeping. The `OR` is the whole fix:

```sql
-- project_2_traverse.py
WHERE e.predicate = $pred AND (e.subject = $anchor OR e.object = $anchor)
```

#### Wall 4: the disambiguation that picked the symptom

The embedding anchor step often returns a *symptom* as the nearest node, not the disease: there are far more symptom
nodes than disease nodes, and "stretchy skin" sits embedding-close to skin-symptom nodes. A symptom is a bad anchor
because the anchor is the node you traverse *from*, and the facts the question wants hang off the disease — its genes, its
inheritance, its own symptoms. Start the walk from a symptom node and you collect facts about the symptom, not the disease
the question is actually about. So I tried to use the graph to break the tie: among the candidates, keep the one connected
by an edge of the picked predicate — figuring the disease I want will sit on that relation and a stray symptom won't.

It failed, and the failure was the lesson. A symptom is the *object* of a `has_phenotype` edge, so it passes the "has an
edge of this predicate" test and gets picked as the anchor. Matching on either side is right for *traversing* (you know
the anchor, you grab the other end) but wrong for *choosing* the anchor — because the thing you're asking about and the
answer sit on opposite ends of the very same edge.

The fix I landed on is almost embarrassingly simple, and I chose it deliberately over a fancier one: **pick the candidate
with the most edges of the relevant predicate — the hub.** A disease has 100+ symptom edges; a single symptom has a
handful. Counting swamps the leaves. No category rules, no direction tables. And it rescued the hard case beautifully: for
"how is the stretchy-skin condition inherited?", the correct disease was ranked *15th* by embedding — buried under skin
symptoms — but it won on edge count and became the anchor.

The whole heuristic is: keep the candidate with the most edges of the picked predicate.

```python
# project_2_disambiguate.py — pick the candidate with the most predicate edges (the hub)
count = edge_count_of(con, cand[0], predicates)   # edges of the picked predicate, either side
if count > best[2]:                               # a leaf has a few; the disease has 100+
    best = (rank, cand, count)
```

That count is taken on *either* side of the edge — subject or object — and that's what keeps the one heuristic general.
Count subject-side only and it would work for `has_phenotype` (where the disease is the subject) but collapse on `causes`,
where the disease is the *object* (gene → disease): the disease would score zero on the very predicate the question
picked. It's the same direction problem as Wall 3, and the same either-side answer. And because this is a *count* rather
than the membership test that just failed, the generality costs nothing — a symptom sits on a `has_phenotype` edge too,
but on a handful, while the disease sits on a hundred, so counting still swamps the leaf.

There's a wrinkle I left in on purpose: the hub is sometimes a well-annotated *subtype* rather than the umbrella EDS node,
so "symptoms of EDS" can answer for one subtype. I wrote it down and moved on, because — see the last project — I didn't
yet have a way to know whether it mattered.

### It works

End to end, the hard question:

```
Q: How is the stretchy-skin condition inherited?
Anchor:    Ehlers-Danlos syndrome, hypermobility type   (recovered from rank 15)
Predicate: has_mode_of_inheritance
Fact:      hypermobility type — has_mode_of_inheritance — Autosomal dominant inheritance
A: ...inherited in an autosomal dominant manner.
```

No disease appeared in the top five embedding hits. The graph recovered it, and the answer is *exactly the traversed
fact*, not a guess.

### Project 3: Hybrid of the previous two methods
I dropped this stage and went straight to project 4.

### Project 4: I stopped, because I could feel the rabbit hole

I skipped the hybrid stage (combining text and graph retrieval) and went straight to measurement, and the reason is a
quote I kept thinking about: *premature optimization is the root of all evil.*

Look at how many knobs I'd accumulated:
 - The similarity cutoff: how wide an angle is too different?
 - How many candidates to pull?
 - The hub heuristic where you identify the anchor by how many instances of the chosen predicate(s) connect to it
(whether as subject or object).
 - Whether to follow one predicate or several and how many?
 - The subtype-vs-umbrella thing. 
Every one is a place I *could* optimize. I had zero evidence which of them was actually costing me answers. 
Tuning any of them would be guessing dressed up as work.

So before touching a single knob, the discipline is a baseline number. And thinking about *how* to measure taught me
several things:

- **There are two separate things to score, and conflating them is fatal.** Retrieval quality (did I pull the right
  facts?) and answer quality (given those facts, is the answer right and grounded?). If you only score the final answer,
  you can't tell a retrieval miss from a generation failure, so you can't fix the right thing.
- **Groundedness has a clean definition and needs no gold answers.** Break the answer into claims, check each against the
  retrieved facts, score the fraction supported. Anything not traceable to the context is a hallucination — even if it's
  true. And because my context is *triples*, I can check a claim by near-lookup instead of fuzzy judgment.
- **The knowledge graph is its own answer key.** For "what genes cause EDS," the correct answer literally *is* the result
  of the `causes` query. I don't need a human to label it. That makes retrieval metrics nearly free — and retrieval is
  exactly where all my heuristics live.

And the thing that made me stop and write all this down: **the traversal side has no natural bottom.** One hop with five
predicates is a fixed template. But multi-hop ("drugs that treat diseases caused by the same gene as EDS") explodes
combinatorially. Move to Neo4j/Cypher or SPARQL and you've bought expressiveness but taken on text-to-query generation as
a hard problem. Involve the ontology's actual semantics — OWL reasoning, inferred edges — and there's a rabbit hole under
the rabbit hole. It's an open research area. The *only* thing that tells you whether any of that depth is worth it is a
number on your own questions.

## The part I didn't expect to learn: how to work with the LLM

I paired with an LLM through all of this, and the working relationship became its own subject. I've previously worked
on projects where I let Claude Code drive, and drive hard and fast. I had a deadline to meet. I reviewed work from a 
high level and directed it to test itself. It worked out the hack-a-thon that used the work. Users accepted and appreciated it. 
The problem is that I didn't know the code as well as if I had written it myself. Since the point of this project
is to learn RAG, I need a more directed approach here. Some rules I started out with include:

- **Drive; don't let it drive.** The rule we settled into: it explains a step, then I choose who takes the keyboard — but
  either way I follow it bit by bit. When I let it run ahead, I got code I didn't understand, which defeats the entire
  point of a learning project.
- **Correct it, in writing, permanently.** We kept a `CLAUDE.md` — a living spec of how I want it to work. When it
  over-explained things I know cold (I had to snap "FFS, I know what an SQL join is"), that became a rule. When it buried
  magic numbers, that became a rule. The corrections compound.
- **It will reach for the robust solution; you may want the simple one.** The sharpest moment: it started building an
  elaborate, "principled" disambiguation with per-predicate config. I stopped it — *this is for learning, not
  production; simpler is more general and a better teaching step.* It immediately switched to the count-the-edges
  heuristic, which is the version above. Left alone, it optimizes for correctness; it needed me to optimize for
  understanding.
- **The failures are the content.** Every wall above — the authoritative definitions, the direction I assumed, the
  disambiguation that grabbed the symptom — came from being made to slow down and check the data instead of trusting the
  first plausible design. That's the whole return on doing it by hand, *or at least slowing down enough to follow closely.*


## What's next, and a bigger thought

Immediate next step is the boring, important one: build the baseline. A small set of EDS questions, gold answers pulled
straight off the graph, and one honest number for how the current pipeline retrieves them. Only then do I get to touch a
knob.

But the leaderboard question was never really what hooked me. What does is that these are different epistemologies — different
theories of how you come to know that two things are related. Text embedding knows the world through language: two entities are
alike if we describe them alike. Network embedding knows it through structure: two nodes are alike if they connect to the rest of
the graph the same way, whatever the words say. And the symbolic traversal in project 2 knows it a third way entirely — through
explicit, typed assertions that some curator chose to write down. Same disease, three ways of knowing, and they don't agree: a
subtype whose description barely overlaps with EDS can be its structural twin, while a node with a rich write-up can sit nearly
alone in the graph. So the comparison I actually want to build next isn't a scoreboard — those exist. It's the one that asks what
each of these ways of knowing can see that the others are blind to. That, more than any single RAG pipeline, is what I came here
for.


