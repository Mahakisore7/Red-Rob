# 02 · Architecture

**Project:** Aptus-R (v5) · **Team:** Code Blooded
All diagrams are Mermaid (render in VS Code with the Mermaid extension and on GitHub).

---

## 2.1 Full pipeline (zoned, two-phase)

```mermaid
flowchart TB

subgraph ZA["PHASE A — Offline precompute (untimed · network OK · run once)"]
  direction TB
  IN["candidates.jsonl.gz (100K) + job_description.md"]
  A1["A1 parse + validate schema<br/>stream one record at a time"]
  A2["A2 skill_index()<br/>merge skills[] + assessment_scores  [G3]"]
  A3["A3 honeypot gate (6 rules)<br/>→ honeypot_ids.json"]
  A4["A4 text_builder<br/>headline+summary+roles+skills(sorted by duration)"]
  A5["A5 embed bge-large-en-v1.5<br/>100K × 1024 float32, normalized"]
  A6["A6 FAISS IndexFlatIP → faiss.index (~3GB)"]
  A7["A7 BM25Okapi → bm25.pkl (~200MB)"]
  A8["A8 feature precompute<br/>S2,S3,S4,S5 precursors + modifiers + penalty flags<br/>→ candidate_features.parquet"]
  A9["A9 facts precompute → candidate_facts.parquet"]
  A10["A10 Weak Ground Truth<br/>~180 hand labels 0–3 → gold_set.csv  [G1]"]
  IN --> A1 --> A2 --> A3 --> A4 --> A5 --> A6
  A5 --> A7
  A2 --> A8 --> A9
  A1 --> A10
end

subgraph ART["ARTIFACTS (committed or download-scripted)"]
  direction LR
  R1["faiss.index"] ~~~ R2["bm25.pkl"] ~~~ R3["*_features.parquet"] ~~~ R4["*_facts.parquet"] ~~~ R5["id_map.json"] ~~~ R6["honeypot_ids.json"] ~~~ R7["jd_embedding.npy"]
end

ZA --> ART

subgraph ZB["PHASE B — rank.py (≤5 min · ≤16GB · CPU · NO NETWORK · deterministic)"]
  direction TB
  B1["B1 load artifacts (~20s)<br/>weights from jd_requirements.yaml  [G2]"]
  B2["B2 embed JD text (local bge)"]
  B3["B3 FAISS search → top-500"]
  B4["B4 BM25 search → top-500"]
  B5["B5 RRF merge → top-500 pool<br/>recall-validated  [G1 fixes F7]"]
  B6["B6 5-signal score × modifiers × penalties × honeypot_mult"]
  B7["B7 LLM rerank top-K (adaptive)<br/>Phi-3-mini · temp=0 · seed · 1-thread  [G4]"]
  B8["B8 blend: w·composite+(1−w)·llm<br/>w EARNED on holdout  [G4 fixes F2]"]
  B9["B9 ranks 31–100 template + grounding validator"]
  B10["B10 CSV + tie-break + assertions + validate_submission.py"]
  B1 --> B2 --> B3 --> B5
  B1 --> B4 --> B5
  B5 --> B6 --> B7 --> B8 --> B9 --> B10
end

ART --> ZB

subgraph ZC["PHASE C — eval.py (untimed)"]
  direction LR
  C1["NDCG@10/@50 · MAP · P@10 · honeypot rate"]
  C2["baselines: naive + title-only (ablation)"]
end

B10 --> ZC
A10 --> ZC

classDef a fill:#f5e6d3,stroke:#c8915b,color:#5a3d1e;
classDef art fill:#fdf3e0,stroke:#d9b46b,color:#6b5320;
classDef b fill:#dceaf7,stroke:#7fa8cc,color:#2e4a6b;
classDef c fill:#e8e3f5,stroke:#8b7fc0,color:#3d2f6b;
classDef g fill:#e0f0e3,stroke:#7fb08a,color:#2e5a38;
class IN,A1,A4,A5,A6,A7,A9 a;
class A2,A3,A8,A10 g;
class R1,R2,R3,R4,R5,R6,R7 art;
class B1,B2,B3,B4,B5,B6,B9,B10 b;
class B7,B8 g;
class C1,C2 c;
```

---

## 2.2 The timing budget (Phase B)

```mermaid
gantt
    title Phase B — 5-minute (300s) budget, adaptive
    dateFormat  X
    axisFormat %s
    section Load
    B1 load artifacts            :0, 20
    section Retrieve
    B2-B5 embed+FAISS+BM25+RRF    :20, 25
    section Score
    B6 score 500 pool            :25, 27
    section LLM (adaptive)
    B7 Phi-3-mini top-K          :27, 147
    section Assemble
    B8-B10 blend+template+write  :147, 157
    section Buffer
    headroom (143s)              :157, 300
```

> **Adaptive rule:** measure the first 3 LLM calls; if projected B7 end > 270s, shrink K (30→20→15→10).
> Never below 10 (NDCG@10 = 50% of score).

---

## 2.3 Scoring formula flow

```mermaid
flowchart LR
  S1["S1 semantic · 0.30"] --> COMP
  S2["S2 career-arc · 0.22"] --> COMP
  S3["S3 behavioral · 0.18"] --> COMP
  S4["S4 recency · 0.15"] --> COMP
  S5["S5 intent · 0.15"] --> COMP
  COMP["composite = Σ wᵢ·Sᵢ"] --> MODS
  MODS["× notice × location × salary × work"] --> PENS
  PENS["× consulting × title-chaser × no-product"] --> HP
  HP["× honeypot_mult (1.0 or 0.05)"] --> FINAL
  FINAL["final_composite"] --> BLEND
  LLM["llm_fit/100 (top-K only)"] --> BLEND
  BLEND["final = w·composite + (1−w)·llm<br/>w earned on holdout"] --> RANK["rank, tie-break by candidate_id"]

  classDef s fill:#e0f0e3,stroke:#7fb08a,color:#2e5a38;
  classDef m fill:#dceaf7,stroke:#7fa8cc,color:#2e4a6b;
  class S1,S2,S3,S4,S5 s;
  class COMP,MODS,PENS,HP,FINAL,LLM,BLEND,RANK m;
```

---

## 2.4 Trap-defeat map

```mermaid
flowchart TB
  subgraph T["The 4 trap classes"]
    T1["Honeypots (~80)"]
    T2["Keyword stuffers"]
    T3["Plain-language Tier-5"]
    T4["Behavioral twins"]
  end
  subgraph D["v5 defences"]
    D1["6-rule integrity gate<br/>+ ×0.05 + assertion"]
    D2["role-coherence S2<br/>+ assessment-contradiction rule"]
    D3["semantic retrieval S1<br/>+ concept thesaurus"]
    D4["recency S4 + intent S5"]
  end
  T1 --> D1
  T2 --> D2
  T3 --> D3
  T4 --> D4

  classDef bad fill:#fde2e2,stroke:#d98b8b,color:#7a2e2e;
  classDef good fill:#e0f0e3,stroke:#7fb08a,color:#2e5a38;
  class T1,T2,T3,T4 bad;
  class D1,D2,D3,D4 good;
```

---

## 2.5 Component / module dependency graph

```mermaid
flowchart LR
  cfg["config.py + jd_requirements.yaml"]
  sch["schema.py (skill_index)"]
  hp["honeypot.py"]
  tb["text_builder.py"]
  emb["embedder.py"]
  ret["retriever.py (FAISS+BM25+RRF)"]
  sig["signals.py (S1–S5+mods+pen)"]
  sco["scorer.py"]
  llm["llm_reranker.py"]
  rea["reasoning.py"]
  out["output_formatter.py"]
  pre["precompute.py"]
  rnk["rank.py"]
  ev["eval.py"]

  cfg --> sch & hp & sig & sco & llm & rea & ret
  sch --> hp & tb & sig & rea
  tb --> emb
  emb --> pre
  hp --> pre
  sig --> sco
  ret --> sco
  sco --> llm --> rea --> out
  pre --> rnk
  rnk --> out
  out --> ev

  classDef core fill:#dceaf7,stroke:#7fa8cc,color:#2e4a6b;
  classDef g fill:#e0f0e3,stroke:#7fb08a,color:#2e5a38;
  class cfg,sch,hp g;
  class tb,emb,ret,sig,sco,llm,rea,out,pre,rnk,ev core;
```

---

## 2.6 Deployment / runtime topology

```mermaid
flowchart TB
  subgraph DEV["Dev machine (Phase A) — network OK"]
    HF["HuggingFace Hub<br/>(bge-large, phi-3-mini GGUF)"] -->|download once| LOCAL["local model cache"]
    LOCAL --> PREC["precompute.py → artifacts/"]
  end
  subgraph JUDGE["Judge sandbox (Phase B) — NO network"]
    ARTC["artifacts/ + models/ (committed/downloaded)"] --> RANKC["rank.py → submission.csv"]
  end
  subgraph CLOUD["HF Spaces (Phase 5)"]
    ST["streamlit_app.py<br/>paste ≤100 JSON → ranked table"]
  end
  PREC -.commit/upload.-> ARTC
  PREC -.same code path.-> ST

  classDef d fill:#f5e6d3,stroke:#c8915b,color:#5a3d1e;
  classDef j fill:#dceaf7,stroke:#7fa8cc,color:#2e4a6b;
  classDef c fill:#e8e3f5,stroke:#8b7fc0,color:#3d2f6b;
  class HF,LOCAL,PREC d;
  class ARTC,RANKC j;
  class ST c;
```

---

## 2.7 Key architectural invariants

1. **The timed step does almost nothing heavy** — it loads artifacts and does matrix math + ≤30 LLM calls.
2. **No network symbol is importable** on the `rank.py` path (offline-test enforced).
3. **Every number is config-driven** — `jd_requirements.yaml` is the single source of truth.
4. **Every artifact length is asserted == 100000** at the end of Phase A.
5. **Determinism by construction** — fixed seeds everywhere; LLM at temp=0, 1 thread.
6. **Graceful degradation** — missing embedding artifact ⇒ retrieval falls back to BM25-only;
   LLM failure ⇒ composite + template reasoning.
