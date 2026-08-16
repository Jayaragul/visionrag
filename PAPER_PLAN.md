# From PRD to Research Paper — Reframing Plan

Source: `vision_rag_product_architecture_roadmap.docx` (v1.0, 16 Aug 2026)
Status: planning. No experiments run yet.

---

## 1. The problem with the current document

It is a product requirements document. It answers *what will be built*.
A paper must answer *what is now known that was not known before*.

Concretely, the PRD is missing all four load-bearing parts of a paper:

- **A claim.** Something that could be false.
- **A baseline.** Something to be better than.
- **A measurement.** On a dataset someone else can obtain.
- **A boundary.** Where the claim stops holding.

Everything below exists to supply those four.

---

## 2. The claim

> Structured event memory — a typed, time-indexed graph built from detection,
> tracking and geometric rules — matches or exceeds caption-based memory on
> temporal and relational video queries, at one to two orders of magnitude
> less compute, and strictly dominates in the CPU-only regime.

Why this claim and not another:

- It is **falsifiable**. If caption-RAG wins at equal compute, the paper dies. Good.
- It does **not require beating SOTA absolutely**. It requires occupying an
  unoccupied point on the accuracy-vs-compute frontier. Winnable solo.
- The PRD already specifies most of the system needed to test it.
- The low-compute regime is genuinely under-studied; nearly all video-RAG work
  assumes a GPU and a large VLM in the ingest loop.

### Secondary claims (each could carry its own paper — do not bundle)

- **S1 — Ego-motion confounding.** Handheld/egocentric event detection is
  systematically confounded by camera motion; explicit compensation plus a
  "could camera motion explain this?" counterfactual gate materially improves
  event precision. (PRD §10.1, §6.3.)
- **S2 — Calibrated abstention.** Evidence-first answering with explicit
  `unknown` states yields better coverage-risk behaviour than forced answering.
  (PRD §5.3, §6.3.)
- **S3 — Cost-aware scheduling.** Load-adaptive detection rate preserves
  accuracy under a CPU budget better than uniform sampling. (PRD §4.3.)

S1 is the strongest standalone spinoff — narrow, tight, and cheap to run.

---

## 3. What survives from the PRD

| PRD section | Fate | Becomes |
|---|---|---|
| §1 Executive summary | Rewrite | Abstract + Intro |
| §1.2 Non-goals | **Keep** | Scope/Limitations — reviewers reward this |
| §2 Phone camera MVP | Cut | One line in Implementation |
| §3 Hardware/software | Compress | Experimental setup (exact CPU, threads) |
| §4.2 C++ pipeline | **Keep** | Method — the ingest path |
| §4.3 Frame scheduling | **Keep** | Method + ablation (S3) |
| §5 Data model | **Keep — this is the core** | Method — the memory substrate |
| §5.2 Event graph | **Keep — this is the core** | Method — Fig. 2 |
| §5.3 Answer contract | **Keep** | Method + calibration eval |
| §6 Deduction engine | Keep layers 1–3, cut 4–5 | Method — relation extraction |
| §6 "Spider-Man" framing | **Delete entirely** | — |
| §7 API contract | Cut | Appendix or repo README |
| §8 Build order | Cut | Repo README |
| §9 Privacy | Compress | Ethics/Broader Impact statement |
| §10 Roadmap phases 3–6 | Cut | Two sentences in Future Work |
| §11 Robotics use cases | Cut | One sentence in Future Work |
| §12 Test strategy | **Rewrite as real evals** | Experiments |
| §13 Milestones | Cut | Repo |
| §14 Key decisions | Compress | Design rationale, ~1 para |
| Appendix A event types | **Keep** | Appendix — the event taxonomy |

Rough survival rate: ~30% of the text, but the *architectural spine* is intact.
The cutting is not loss — a paper that claims one thing and proves it beats a
paper that describes six phases and proves nothing.

---

## 4. Target paper skeleton

**Working title** (avoid "Embodied-RAG" / "Video-RAG" — both taken):
something along the lines of *"Event-Graph Memory: Compute-Efficient Grounded
Question Answering over Long Video"*. Decide after the literature check.

1. **Abstract** — claim, method, headline number, headline cost number.
2. **Introduction** — caption-based memory is accurate and expensive; many
   deployment settings have no GPU; is the captioning step actually necessary?
3. **Related work** — video QA/RAG; episodic & robot memory; dynamic scene
   graphs; efficient inference. *Must be current — see §7.*
4. **Method**
   - 4.1 Perception ingest (detect → track → ego-motion → geometry)
   - 4.2 Event induction rules and the typed event schema
   - 4.3 Memory: structured store + embedding index (why both)
   - 4.4 Retrieval and evidence-grounded answering with abstention
5. **Experimental setup** — dataset, baselines, exact hardware, compute accounting.
6. **Results** — E1–E5 below.
7. **Limitations** — lift directly from PRD §1.2. Say plainly what fails:
   fine-grained actions, intent, unmodelled event types, metric speed.
8. **Ethics / broader impact** — from PRD §9. Surveillance-adjacent work needs
   this handled seriously, not as boilerplate.

---

## 5. Experiments — the minimum credible set

Do not start writing before E1, E2 and E5 exist.

### E1 — Main result: accuracy vs. compute
- X-axis: CPU-seconds per minute of video ingested (log scale). Y-axis: task accuracy.
- **Baselines (non-negotiable):**
  - B1: VLM-captioning RAG (uniform sampling → caption → embed → retrieve).
    This is the one that matters. API-based is fine; report the cost.
  - B2: Uniform frame sampling straight into a VLM, no memory.
  - B3: Retrieval-free — question to LLM with no video. Establishes the floor
    and catches language-prior leakage in the benchmark.
- Sweep your system across scheduler budgets to trace a curve, not a point.

### E2 — Memory ablation
Detections only → +tracks → +events → +relations.
Answers "which part of the structure is doing the work?" Reviewers always ask.

### E3 — Ego-motion ablation (claim S1)
Event precision/recall with and without global motion compensation, split by
fixed vs. handheld footage. Report the counterfactual gate separately.

### E4 — Calibration and abstention
Grounding rate (does each answer statement trace to a stored event/frame?),
coverage-risk curve, ECE. Needs a hand-labelled subset — budget for it.

### E5 — Cost characterisation
Latency p50/p95, peak RSS, throughput. **Name the CPU, core count, thread
count, and whether it was thermally throttled.** A "CPU-only" claim with
unnamed hardware is not a claim.

### Benchmarks to evaluate
Prefer an existing one — building a dataset roughly triples the timeline.
Candidates: Ego4D NLQ, QAEgo4D, EgoSchema, NExT-QA, ActivityNet-QA.
Check licences and download sizes before committing; Ego4D in particular has
an access agreement and is large.

**Selection criteria:** the benchmark must contain genuinely *temporal and
relational* questions. If accuracy can be had from a single frame, it cannot
test this claim.

---

## 6. Realistic timeline (part-time, solo)

| Phase | Work | Duration |
|---|---|---|
| 0 | Literature check; confirm novelty; pick benchmark | 2 weeks |
| 1 | Ingest path: detect → track → events → store | 4–6 weeks |
| 2 | Retrieval + grounded answering + abstention | 2–3 weeks |
| 3 | Baselines (B1 is the expensive one) | 2–3 weeks |
| 4 | E1–E5, plus the inevitable re-runs | 3–4 weeks |
| 5 | Writing, figures, repo cleanup | 3–4 weeks |

**≈ 3–5 months to a credible preprint.** Phase 3 is the most commonly
underestimated: building a *fair* baseline is real work, and an unfair baseline
is the fastest way to get a paper dismissed.

---

## 7. Before committing: the literature check

This determines whether the claim is novel and must happen first.

Known-adjacent lines of work to check thoroughly (this list reflects knowledge
to ~mid-2026 and is **certainly incomplete** — verify against current arXiv):

- Video RAG / long-video QA: agentic and retrieval-based video QA systems
- Episodic and robot memory: ReMEmbR, Embodied-RAG, ConceptGraphs, HOV-SG
- Dynamic scene graph generation: Action Genome and successors
- Efficient/edge video understanding; token- and frame-reduction methods

**Outcome to look for:** someone may already have done caption-free structured
video memory. If so, the claim shifts to the *compute frontier* and the
*ego-motion counterfactual*, which are still likely open.

Finding prior work is not failure — finding it *after* four months of
implementation is.

---

## 8. Publication and credibility path

1. **Repo first, and public.** Reproducible code with the exact commit used for
   the numbers. This carries more weight than the PDF for a first-time author.
2. **arXiv preprint.** Note: a first submission to `cs.CV` / `cs.RO` requires
   **endorsement** from an existing author in that category. Arrange early —
   it surprises people at the worst moment.
3. **Workshop submission.** This is the step that converts a preprint into
   reviewed work. Target CVPR/ICCV/ECCV workshops on efficient or egocentric
   video, or ICRA/IROS workshops on robot perception and memory.
   Do not aim a first paper at a main conference.
4. **Then** consider a main-track or journal version with the extra claims.

An unreviewed preprint is a weak credential on its own. Preprint + working code
+ workshop review is a strong one.

---

## 9. Open decisions

- [ ] Which claim leads: main (compute frontier) or S1 (ego-motion)?
- [ ] Which benchmark, and is its access agreement acceptable?
- [ ] Is a GPU available for the B1 baseline, or is it API-based?
- [ ] Detector/tracker choice — must be pinned and version-locked for repro.
- [ ] New name, once the literature check clears.
