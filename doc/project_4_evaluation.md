# Project 4: Apply / Evaluate (design notes)

Design notes, not a built system yet. This captures the reasoning about *how* to measure projects 1 and 2 before any
eval harness is written — deliberately, because the measurement decisions shape what's worth building.

## The problem and the approach

Projects 1 and 2 each end in a working system, but "working" so far means "produces a grounded-looking answer on a few
hand-typed questions." That is not measurement. Both systems are full of knobs and heuristics — the similarity cutoff, the
recall depth `k`, the degree heuristic for disambiguation, umbrella-vs-subtype anchoring, following one predicate vs the
whole set — and right now there is no evidence which, if any, is costing us answers. Tuning any of them today is guessing.

So the approach is metrics first, optimization second. A metric turns "this feels loose" into "this knob moves the number
by X," which is the only honest signal that a heuristic is worth the complexity it adds. The discipline is a **baseline**:
one number for how the current pipeline does, which you refuse to optimize against until it exists. *Premature
optimization* here isn't a slogan — the traversal side (below) has no natural bottom, so without metrics you can
disappear into it.

## Two separable things to measure

A RAG system has two things to score, and keeping them apart is the whole game:

1. **Retrieval quality** — did we pull the *right facts*? Judged on the retrieved context alone, before the generator
   runs. Project 1: did nearest-node search surface the relevant node texts? Project 2: did we anchor on the right node,
   pick the right predicate, and traverse to the right edges?
2. **Answer quality** — given whatever context was retrieved, is the final answer correct, *grounded* (faithful to the
   context, no hallucination), and complete?

They matter separately because a bad answer has two very different causes: a **retrieval miss** (good generator, wrong
context) or a **generation failure** (good context, wrong answer). Score only the final answer and you cannot tell which,
so you cannot fix the right thing. Scoring retrieval and generation apart localizes the fault.

This also maps onto the two heuristic-heavy halves of project 2. The **ontology** side — predicate menu, hand-written
descriptions, the cutoff — is "did we understand what the question asks of the schema?" The **navigation** side — anchor,
disambiguation, traversal, direction — is "did we get to the right facts?" Both become measurable the moment we have
graph-derived gold answers.

## Measuring groundedness (faithfulness)

Groundedness has a compact definition: **break the answer into individual claims, check each claim against the retrieved
context, and score the fraction the context supports.** A claim you can't trace back to the context is a hallucination —
even if it happens to be true, because it came from the model's memory, not your source.

The one moving part is *who does the checking*:

- **LLM-as-judge / NLI:** ask a model "does this context entail this claim?" for each claim. This is what general RAG eval
  tools do, because plain-text context can't be matched literally.
- **For KG-RAG, better:** the context is *triples*, so a claim can be checked against the actual facts more exactly.
  "B3GALT6 causes EDS spondylodysplastic type 2" → is that triple literally in the retrieved set? Structured context
  turns grounding closer to a lookup than a fuzzy judgment.

Crucially, faithfulness measures only whether the answer *stays within the context*. It says nothing about whether the
answer is correct or complete — that is the retrieval axis and the gold set. An answer can be perfectly grounded and still
wrong, if retrieval fed it the wrong facts. And notably, **groundedness needs no gold standard at all**: it is
answer-vs-context, self-contained.

## How much human curation does gold really need?

Less than it first appears, and the distinction is worth being precise about:

- **Grounding needs no gold** (above).
- **Correctness/completeness needs *a reference*, but not necessarily hand-labeled answers.** For structured questions the
  reference is *derived* from the graph: the answer to "what genes cause EDS" is literally the result set of the `causes`
  query. Nobody labels the answer. The catch is that someone must assert "*this* question corresponds to *that* query,"
  and that is a little circular — if you can write the gold query, you have essentially solved the task for that question.
  So curation shifts from *labeling answers* (expensive) to *authoring question↔query pairs* (cheaper, not zero).
- **Synthetic generation pushes it further down.** Start from a known triple or subgraph, generate a question *from* it
  (template or LLM), and the gold answer is the triple you started with. That is how much of KGQA benchmarking scales
  without per-answer labeling. The cost: synthetic questions may not sound like real user phrasing, so you are testing
  against your own distribution.

So a human defines *what correct means* somewhere, but for a KG that can be an assertion about queries, or a generator —
not a pile of hand-written answers.

### The KG as its own answer key

The favorable property that makes the above cheap for us: for project 2's structured questions, **the graph is both the
retrieval source and the answer key.** "What genes cause EDS?" has ground truth sitting right there — the `causes` /
`gene_associated_with_condition` edges out of the EDS node. That means retrieval metrics (precision/recall of the pulled
facts against the graph's own edges) are nearly free, and retrieval is exactly where all the navigation heuristics live.
Project 1 (plain vector retrieval over node text) does not get this for free — "relevant" is fuzzier there.

## How deep the traversal rabbit hole goes (and why the metrics matter)

The traversal side is effectively unbounded — it is an open research area (KGQA / semantic parsing) — and it deepens along
three axes at once:

1. **Hops.** One hop with five predicates is a fixed template. "What drugs treat diseases caused by the same gene as EDS?"
   is three or four hops, and the number of candidate paths grows combinatorially with hop count × predicate count.
   Choosing *which* path answers the question is the hard part.
2. **Query language.** Neo4j/Cypher and SPARQL are not the rabbit hole — they are the tools built to manage it
   (variable-length paths, patterns miserable to express in SQL). But adopting them makes *text-to-Cypher /
   text-to-SPARQL* central, and reliably getting an LLM to emit correct queries — and validating what it emits — is its
   own deep problem.
3. **The ontology itself.** Once OWL/RDFS semantics are involved, do you traverse only asserted edges or *inferred* ones
   (via subclass/subproperty reasoning)? Reasoning returns implied facts a plain traversal misses, and brings open-world
   semantics, decidability, and performance costs — a rabbit hole under the rabbit hole.

This is the whole reason the metrics-first instinct holds: with no natural bottom, the only thing that tells you whether
multi-hop, or text-to-SPARQL, or reasoning is *worth it* is whether it moves the number on your questions. Plenty of real
EDS questions are one-hop; a metric tells you what fraction actually need the deeper machinery, so complexity is spent only
where it pays.

## Not yet (the first concrete step)

- A small set of structured EDS questions whose gold answers come straight from the graph (question↔query pairs).
- A retrieval baseline: precision/recall of project 2's pulled facts against that gold, before any knob is touched.
- Later: a faithfulness check (triple-matching for KG-RAG), and only then targeted optimization of whichever knob the
  numbers implicate.
