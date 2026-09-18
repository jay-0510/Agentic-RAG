# Agentic RAG

This document explains the project in plain language — no jargon walls — so you can describe it in an interview, to a non-technical person, or to a college student who's never heard of RAG.

---

## 1. Start with the basic idea: RAG

Imagine you ask a very well-read friend a question. If it's general knowledge, they answer from memory. But if you ask something specific — "what's on page 12 of this report you've never seen?" — a smart friend says "let me check the document first," reads the relevant part, and then answers based on what they just read.

That's **RAG (Retrieval-Augmented Generation)**:

- **Retrieval** = go look something up in an external source (a database, a set of documents, an API).
- **Augmented Generation** = hand what you found to the AI so it answers using those facts, instead of guessing from memory.

Why bother? Because an LLM only knows what it was trained on, up to a certain date, and it can't know your private documents, your company's data, or anything that changed after training. RAG plugs in fresh, specific, or private knowledge at answer-time.

**Traditional RAG is a fixed recipe:** question comes in → always go look something up → always hand it to the AI → always return whatever it says. It never questions any of those steps.

## 2. Now add the "agentic" part

A traditional RAG system is like a junior employee following a checklist no matter what: "Step 1: search. Step 2: summarize. Step 3: reply." Even if the question didn't need a search, or the search returned junk, the checklist runs anyway.

An **agentic** system replaces the checklist with a decision-maker — an "agent" that reasons about what to do at each step, the way a competent human researcher would:

- _"Do I actually need to look this up, or do I already know the answer?"_
- _"Is my search phrased well, or should I rephrase it before searching?"_
- _"Did what I found actually answer the question, or was it junk — should I search again differently?"_
- _"Before I answer, does my answer actually match what I found, or am I making something up?"_

**Agentic RAG = RAG + an agent that makes these decisions dynamically**, instead of blindly always retrieving once and generating.

So the two building blocks are:

1. **The autonomous agent** — the decision-maker/planner. In this project, it's built with a framework called LangGraph, which lets you describe the workflow as a flowchart with branches and loops, and lets the AI's own reasoning decide which branch to take.
2. **RAG (the knowledge source)** — this can be a vector database, a plain database, an external API, or (as in this project) a simple local JSON file of sports facts. It's whatever the agent calls when it decides retrieval is needed.

## 3. What this specific project does

It's a small "Sports Knowledge Service" — you ask it a sports rules question (e.g., "when is a player offside in football?"), and instead of just doing one fixed retrieve-then-answer pass, it works through a small pipeline of decisions:

1. **Decide if a lookup is needed at all.**
2. **If yes, rewrite the question into a good search query** (people ask messy, ambiguous questions; a cleaner query retrieves better).
3. **Retrieve candidate facts** from the knowledge source.
4. **Grade whether those facts are actually relevant.** If they're weak or off-topic, go back and rewrite the query, then try again — up to a limited number of retries, so it doesn't loop forever.
5. **Generate a draft answer** using only the retrieved facts as grounding.
6. **Verify the draft is actually faithful to those facts** — i.e., check the AI isn't inventing claims that aren't supported. If it's not well-grounded, it can loop back for another, stricter retrieval pass.
7. **Return the final answer.**

That's it — a loop that can rewrite its own search, discard bad results, retry, and double-check its own answer before handing it back.

## 4. Traditional RAG vs. Agentic RAG — the one-sentence version

> **Traditional RAG always follows the same steps once, no matter what. Agentic RAG makes a decision at each step, and can retry or self-correct when something looks wrong.**

A slightly longer version, useful for an interview answer:

- Traditional RAG is a **straight line**: retrieve → generate. It has no way to notice or recover from a bad retrieval or a bad answer.
- Agentic RAG is a **loop with checkpoints**: it can decide not to retrieve at all, rewrite a bad query, discard irrelevant results and try again, and double-check its own final answer before returning it — all driven by the model's own reasoning, not a hardcoded script.

## 5. When should you actually use Agentic RAG?

It's not always the better choice — it's a trade-off.

**Use Agentic RAG when:**

- Questions are often vague, ambiguous, or phrased in many different ways (so a single fixed search often misses).
- Your knowledge base is noisy, large, or inconsistent, so a single retrieval attempt isn't reliably good enough.
- Wrong or unsupported answers are costly (e.g., customer-facing or decision-support tools), so the verification/self-check step earns its cost.
- You can tolerate a bit more latency and a few extra LLM calls per question.

**Stick with traditional (plain) RAG when:**

- The knowledge base is small, clean, and well-scoped (questions map cleanly to a known set of facts).
- Latency and cost matter more than squeezing out the last bit of accuracy.
- You want something simple to build, debug, and explain — a fixed pipeline is much easier to reason about than a looping agent.

## 6. Remember three things

1. **RAG** = look something up, then let the AI answer using what it found.
2. **Agentic** = replace the fixed "always look up once" script with a reasoning agent that decides _whether_, _what_, _where_, and _how many times_ to look things up — and checks its own work before answering.
3. **This project** is a small, concrete example of that: a router, a query rewriter, a retriever, a relevance grader with retries, a generator, and a faithfulness verifier, all wired together as a LangGraph state machine.
