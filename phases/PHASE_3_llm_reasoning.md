# Phase 3 · Local LLM Rerank + Reasoning

**Owner:** RamKumar KR · **Duration:** ~Day 3 · **Tag:** `v0.4-llm`
**Goal:** Add the Phi-3-mini rerank for the top-K and the grounded reasoning generator — both
**deterministic** and **time-bounded**. This is where v4's three LLM flaws (F2/F3/F8) are closed.

---

## 0. Principle
The LLM is additive and on probation. It enters the score only if it earns its weight (Phase 4,
Decision Rule 1) and only within a measured time budget (Decision Rule 2). It always degrades
gracefully.

---

## 1. `llm_reranker.py`

### 1.1 Model load (offline, deterministic — fixes F8)
```python
from llama_cpp import Llama
llm = Llama(model_path="models/phi-3-mini-q4.gguf",
            n_ctx=2048, n_threads=1, seed=42, verbose=False)   # 1 thread + seed = reproducible
# generation: temperature=0.0, top_p=1.0, max_tokens=180
```

### 1.2 Prompt (real fields only — no hallucination possible in the prompt)
```
You are a senior technical recruiter for a Senior AI Engineer role at an AI startup
(Pune/Noida, hybrid, 5–9 years). Ideal: shipped production retrieval/ranking/recommendation
systems, strong Python, vector databases, embeddings, product-company background.

Candidate ID: {cid}
Title: {title} | Company: {company} | Experience: {yoe} years
Location: {location} | Relocate: {relocate}
Notice: {notice} days | Salary: {smin}-{smax} LPA
Recent roles: {role1}; {role2}; {role3}
Top skills: {skill1} ({prof1}, {mo1}mo), {skill2} (...), {skill3} (...)
GitHub: {github} | Active: {last_active} | Open to work: {otw}

Respond with valid JSON only:
{"fit_score": <0-100>, "hire_recommendation": "<strong_yes|yes|maybe|no>",
 "reasoning": "<=2 sentences: cite specific facts, connect to JD, note the biggest concern>"}
```
Skills sorted by `duration_months` desc (deepest first).

### 1.3 Adaptive timing gate (fixes F3 — Decision Rule 2)
```python
times = []
for i, cand in enumerate(top_pool):
    t0 = time.perf_counter(); out = llm_call(cand); times.append(time.perf_counter()-t0)
    if i == 2:                                  # after 3 calls, project
        mean = sum(times)/len(times)
        budget_left = 290 - elapsed_since_start()
        K = max(10, min(K, int((budget_left - 15) / mean)))   # never below 10
```
Shrink 30→20→15→10 if projected to overrun. Log the chosen K.

### 1.4 Malformed-JSON fallback (FR-17)
```python
try:
    j = json.loads(out)
    fit = float(j["fit_score"]); reason = j["reasoning"]
except Exception:
    fit = composite[cid]*100; reason = template_reason(cid); log_llm_failure(cid)
```

### 1.5 Blend (weight decided in Phase 4)
```python
final[cid] = w*composite[cid] + (1-w)*(fit/100)   # w from config, default chosen by Decision Rule 1
```

---

## 2. `reasoning.py`

### 2.1 Top-K: LLM reasoning
Use the LLM's `reasoning` field, after the grounding validator (below).

### 2.2 Ranks (K+1)–100: template reasoning (varied, grounded)
Scenario-matched templates pulling real fields from `candidate_facts.parquet`:
| Scenario | Trigger | Template skeleton |
|---|---|---|
| good fit, short notice | high composite, notice ≤30 | "{title} with {yoe}y at product-company building {skill}; {notice}-day notice and active GitHub ({gh}) make this actionable." |
| good skills, long notice | high composite, notice >60 | "{title} with {yoe}y and strong {concept} background ({skills}); {notice}-day notice is a timeline concern." |
| consulting background | is_consulting_only | "{title} with {yoe}y depth in {concept}; career is consulting-firm based, a weak signal for product ownership." |
| location mismatch | international/no-relocate | "{title} with strong {concept} skills and {yoe}y; based in {location} with relocate={relocate}, friction for this hybrid role." |
Template index = `hash(candidate_id) % n_variants` → variety; deterministic.

### 2.3 Grounding validator (FR-19 — zero hallucination by construction)
For every filled slot, re-read the field from the candidate record and assert equality. On mismatch,
**raise** (don't emit). For LLM reasoning, verify any number/skill it names appears in the record;
if it invents a fact, fall back to template. This is the hard guarantee.

### 2.4 Tone by rank
ranks 1–10 assertive · 11–50 positive+caveat · 51–100 conditional.

---

## 3. Determinism finalization (fixes F8, NFR-5)
- `PYTHONHASHSEED=0`; numpy seed fixed; LLM `seed=42`, `n_threads=1`, `temperature=0`.
- `tests/test_determinism.py`: run `rank.py` twice → byte-identical CSV.

---

## 4. Full-pipeline timing dry-run (fixes F3)
Run `rank.py` end-to-end with the LLM, on a **throttled** CPU (e.g. limit threads) to simulate a weak
judge box. Confirm ≤5 min with the adaptive gate engaged. Record the timing table.

---

## 5. Deliverables
- [ ] `llm_reranker.py` — deterministic decode, adaptive gate, JSON fallback
- [ ] `reasoning.py` — LLM (top-K) + template (rest) + grounding validator + tone
- [ ] `tests/test_determinism.py` green
- [ ] full pipeline < 5 min on throttled CPU; timing table recorded
- [ ] top-10 reasoning manually inspected: specific, JD-connected, honest concerns, varied

---

## 6. Definition of Done
- Full `rank.py` (with LLM) runs < 5 min, deterministic, valid CSV.
- Grounding validator passes; no reasoning contains an unverifiable fact.
- Fallback path tested (force a malformed JSON, confirm graceful template).

**Git:** merge `phase-3` → `main`, tag `v0.4-llm`.

> **Note:** the LLM blend weight `w` is still provisional here. It is *finalized by measurement* in
> Phase 4. If Phase 4 says the LLM doesn't help, `w→1.0` and the LLM stays only as a reasoning writer.

---

## 7. Risks
| Risk | Mitigation |
|---|---|
| Phi-3 weak JSON adherence | strict prompt + fallback; consider mistral-7b-q4 (declared fallback) |
| LLM slower on judge HW | adaptive gate, never below K=10 |
| LLM invents facts | grounding validator forces template fallback |
| Non-determinism | seed + 1 thread + temp=0; determinism test gates the tag |
