# Gemini Pro Deep Research — Mega-Prompt (ESORICS / LNCS)

> Paste this entire document as the instruction prompt. Attach:
> - the Phase 3 experimentation PDF,
> - `FIGURE_CAPTIONS_PHASE3.md`,
> - the **six** figures listed below,
> - and a **required architecture-diagram placeholder note** (since the architecture figure is not attached yet).

You are writing a **single-blind ESORICS full paper** in **LNCS** style. You will produce a **professional, security-first** paper draft that is fully consistent with the attached Phase 3 experimentation document and the attached figure pack. Do **not** invent results, do **not** contradict the figures, and do **not** use hype language.

## Submission constraints (LNCS / ESORICS — mandatory)
- **Template compliance at submission**: the paper must comply with the official LNCS template at the time of submission (10-point font; do not alter margins).
- **Length**: at most **16 pages** excluding bibliography and clearly marked appendices; at most **20 pages total**. The paper must be intelligible without appendices.
- **Language**: English.
- **Review model**: **single-blind** (authors and affiliations are disclosed; reviewers are anonymous). Do not write “anonymous submission” language.
- Submissions not meeting these guidelines risk desk rejection. Treat these as hard constraints while drafting.

---

## Inputs you will be given (treat as authoritative)

1. **Phase 3 experimentation document (authoritative reference)** as a **single PDF**.
   Numeric results and system description must match this document exactly.

2. **Figure pack (six images — exactly this order and labeling)**

   | Fig | File | Role |
   |---|---|---|
   | Fig. 1 | `fig_apb_heatmap.pdf` (generated) | **Primary results figure (APB-F1)** |
   | Fig. 2 | `fig_macro_heatmap.pdf` (generated) | Macro-F1 (secondary metric) |
   | Fig. 3 | `fig_sleeper_fingerprint.pdf` (generated) | Sleeper-attack dynamics |
   | Fig. 4 | `fig_apb_with_ldm.pdf` (generated) | Synthesis (worst-case + diffusion) |
   | Fig. 5 | Existing image: APB/Macro vs round | Diffusion time-series |
   | Fig. 6 | Existing image: BSR vs round | Security metric (BSR) |

3. **Figure captions + provenance**: `docs/FIGURE_CAPTIONS_PHASE3.md`. Use the captions and Alt Text from this file; do not paraphrase them destructively.

4. **Architecture diagram placeholder** (not attached):
   - You must include a short **Architecture overview** subsection in the paper text that is fully intelligible without the figure.
   - You must insert this placeholder sentence verbatim near the end of that subsection:
     - **“Figure X (architecture) will depict the orchestration cycle and the trust-gated augmentation pipeline.”**
   - Do not refer to any non-existent figure numbers elsewhere in the paper.
   - The architecture figure will be provided later by the authors and inserted during final LaTeX assembly.

---

## What the paper is about (two linked ESORICS contributions)

### (A) Primary quantitative contribution — clinical safety under attack
Rare-class preservation under adversarial FL: **APB (Class 3) F1 remains ≥ 0.75 across all three attack models only for PoC**; PoC is the best APB defense under the sleeper attack; trust-gated LDM further improves APB-F1 over the strongest non-ledger baseline.

### (B) Co-primary security contribution — accountable governance
The ledger is **not passive logging**. It provides immutable provenance for both aggregation and augmentation decisions, enabling **accountable trust-gated diffusion** and **post-hoc forensics**, and closes a distinct **augmentation stealth channel** that exists when diffusion is ungated.

---

## Threat model (must match Phase 3 doc exactly)

1. **Gaussian weight poisoning** — attackers upload random weights.
2. **Semantic label-flip** — attackers flip labels before local training.
3. **Sleeper / backdoor-critical-layer label-flip**, **activated at Round 15**, inspired by:
   - Foroughi et al., arXiv:2602.15161 (2026).
   - Zhuang et al., arXiv:2308.04466 (2023).
4. **Augmentation stealth bypass under diffusion** — blind diffusion creates a separate poisoning channel via synthetic requests; trust-gated diffusion blocks this channel post-activation.

---

## Methods (must match Phase 3 doc exactly)

Defenses: FedAvg, Krum, Multi-Krum, Median, Trimmed Mean, Bulyan, and PoC-based selection. LDM (class-conditional 1D UNet diffusion) is used only where the Phase 3 doc uses it (diffusion comparison).

---

## Metrics

- **Primary**: APB-F1 (Class 3) — clinical priority metric.
- **Secondary**: macro-F1 — framed explicitly as secondary; show why it can mislead.
- **Security (diffusion setting)**: Backdoor Success Rate (BSR).
- Latency is **not a figure**; mention briefly only in Experimental Setup if needed.

---

## Numeric claims you are allowed to make (use exactly these)

- Gaussian setting: PoC macro-F1 **0.9645**, APB-F1 **0.8773**; FedAvg macro-F1 **0.1711**, APB-F1 **0.0000** (Fig. 1, Fig. 2).
- Label-flip setting: FedAvg macro-F1 **0.9613**; PoC macro-F1 **0.9586**, APB-F1 **0.8418** (Fig. 1, Fig. 2).
- Sleeper setting: PoC APB-F1 **0.7858** (best); Krum APB-F1 **0.3065** (catastrophic). Both visible on Fig. 1 and Fig. 3.
- Diffusion setting (sleeper + LDM): PoC + Trust-Gated LDM final APB-F1 **0.8785**; Multi-Krum + Blind LDM final APB-F1 **0.8568** (Fig. 4, Fig. 5). Final-round BSR: PoC + Trust-Gated LDM < Multi-Krum + Blind LDM (Fig. 6).

If a number is not listed above or not visible in a figure/table, do **not** use it.

---

## Required output format (what you must produce)

Produce the following, in this order. Do not include filler content.

### 1. Glossary and notation table
- Terms: PoC, Active-Ledger, Trust-Gated LDM, APB, BSR, macro-F1, MIT-BIH, AAMI.
- Symbols: α (Dirichlet concentration), **γ** (EMA smoothing, **never α**), T (diffusion timesteps), f (assumed Byzantine count).

### 2. Claims → Evidence map
Every numeric/qualitative claim must be listed with:
- Claim text (1–2 sentences).
- Figure/table reference (Fig. 1–6 from the table above).
- Exact `.npy` field or Phase-3 doc section used.

### 3. Paper outline with page budget (≤16 pages before references)
LNCS section list, with an approximate word budget per section.

### 4. Full draft paper in **complete LaTeX (LNCS)**
You must output a single, self-contained LaTeX manuscript that compiles under the LNCS template:
- Use `\\documentclass[runningheads]{llncs}`.
- Include required packages only (keep it minimal).
- Include `\\title{}`, full `\\author{}` block with affiliations, and `\\institute{}`.
- Include `\\begin{abstract}` and `\\keywords{}`.
- Include all sections and subsections.
- Include figure environments with `\\includegraphics{}` placeholders matching the attached figure filenames (Fig. 1–6), and use captions consistent with `FIGURE_CAPTIONS_PHASE3.md`.
- Add a placeholder sentence in the Architecture overview as specified (Figure X placeholder).
- Include a references section (BibTeX or `thebibliography`), with complete entries and no uncited references.

Required sections and minimum content:
- **Introduction** — clinical motivation, adversarial FL, contributions (A) and (B), roadmap.
- **Background and Related Work** — blockchain-FL (explicitly cite contribution-aware selection and committee-style coordination, and state the delta precisely; do **not** claim prior work is only “passive logging”), robust aggregation, backdoor/sleeper FL, synthetic augmentation, clinical ECG FL.
- **System Overview** — Active-Ledger: ledger (events), PoC score (with γ), trust-gated diffusion; include an architecture placeholder sentence (no architecture figure is attached yet).
- **Threat Model** — Gaussian, semantic label-flip, sleeper, stealth augmentation; do not depend on an architecture figure.
- **Methods** — defenses, PoC scoring equation using γ, diffusion gating, evaluation pipeline.
- **Experimental Setup** — MIT-BIH, AAMI mapping, non-IID Dirichlet α = 0.5, 10 clients (2 Byzantine), 40 rounds, metrics.
- **Results** — use this figure order:
   - **Fig. 1** (APB-F1 matrix) as the headline result.
   - **Fig. 2** (macro-F1 matrix) to contrast.
   - **Fig. 3** (sleeper fingerprint) for sleeper dynamics.
   - **Fig. 4** (worst-case + diffusion) as the synthesis figure.
   - **Fig. 5** (APB/Macro time-series) for diffusion training dynamics.
   - **Fig. 6** (BSR) for the security metric.
- **Discussion / Limitations** — why FedAvg may look good on macro-F1 (dilution), why geometry fails for sleeper, auditability and compliance value, and explicit limitations (single-seed runs, 20% attacker ratio, etc.).
- **Conclusion**.

### 5. References checklist
List every citation key used, with complete bibliographic info (authors, title, venue, year, DOI/URL). Flag uncertain fields as “to verify”; never guess a DOI or venue.

---

## Image placement and interpretation guide (hard)

For each figure you cite, follow the placement and interpretation rules below. Do not describe behavior that is not visible in the figure.

### Fig. 1 — APB-F1 robustness matrix (primary results figure)
- **Place**: First figure of the Results section.
- **Reference in**: Results, Discussion.
- **Interpret**: PoC is the only defense whose APB-F1 stays above the 0.75 clinical line across all three attacks. FedAvg collapses under Gaussian. Krum collapses under the sleeper attack. Cite exact values (Section *Numeric claims*).
- **Do not**: rank methods by macro-F1 here; that is Fig. 3.

### Fig. 2 — Macro-F1 robustness matrix (secondary metric)
- **Place**: Immediately after Fig. 2.
- **Reference in**: Results (short paragraph), Discussion (longer paragraph on dilution).
- **Interpret**: FedAvg appears to win macro-F1 on label-flip and sleeper due to dilution from 20% attacker ratio + non-IID honest gradients, but Fig. 2 shows FedAvg is not clinically safe. Use this figure to justify why the paper uses APB-F1 as the primary metric.
- **Do not**: contradict Fig. 2 rankings.

### Fig. 3 — Sleeper-attack fingerprint
- **Place**: In the sleeper-attack results subsection.
- **Reference in**: Results.
- **Interpret**: Show that (i) macro-F1 does not reveal Krum’s failure, (ii) APB-F1 clearly separates PoC from Krum after round 15. Point to the 0.75 horizontal clinical line and the vertical round-15 activation line.
- **Do not**: discuss methods not plotted (Median / TrimmedMean / Bulyan) as if they are in this figure.

### Fig. 4 — Worst-case APB-F1 + diffusion (synthesis)
- **Place**: End of Results (before Discussion).
- **Reference in**: Results synthesis paragraph, and as the bridge to diffusion results.
- **Interpret**: Left block is the worst-case across the three attacks; right block is the diffusion comparison. PoC + Trust-Gated LDM (ours) is highest; Multi-Krum + Blind LDM is close, but Fig. 6 shows it is worse on BSR — the ledger provides an additional security guarantee that bars alone do not capture.
- **Do not**: present the right-block comparison as macro-F1; it is APB-F1.

### Fig. 5 — APB/Macro vs round (existing image)
- **Place**: Diffusion subsection, before Fig. 6.
- **Reference in**: Results.
- **Interpret**: Training dynamics; both methods stay above the 0.75 clinical threshold on APB after early rounds; macro-F1 converges above 0.95. The attack-activation vertical line does not produce a visible collapse because the honest-pool + gating suppress the attack.
- **Do not**: imply convergence numbers not in the plot.

### Fig. 6 — BSR vs round (existing image)
- **Place**: Immediately after Fig. 6.
- **Reference in**: Results security-metric paragraph.
- **Interpret**: Both methods produce non-zero BSR after round 15. The final-round BSR for Multi-Krum + Blind LDM is higher than for PoC + Trust-Gated LDM — ungated augmentation admits poisoned synthetic requests while the ledger gate (Fig. 1 Panel B) suppresses them.
- **Do not**: claim BSR is zero for either method.

---

## Style and hygiene rules (strict)

- **Abstract**: 150–250 words, **at most 1–2 numbers**, structure:
  Problem → Threat model → Method (PoC + ledger + trust-gated diffusion) → Key result → Security/auditability contribution.
- **Notation**: Dirichlet concentration is **α** (e.g., α = 0.5). EMA constant is **γ** (e.g., γ = 0.7). Do **not** reuse α for EMA.
- **Terminology**: PoC is **not** a cryptographic proof or consensus. Define it as an **on-chain, history-based contribution score** for client selection and synthetic-data gating. You may use “Contribution Score” in prose; keep PoC as the acronym if used.
- **Causal attribution**: Separate security motivations from engineering (gas/runtime) motivations. Do not conflate them in a single causal sentence.
- **Tone**: No rhetorical phrases like “crushing”, “total collapse”, “mathematically crushing”. Use measurable statements only.
- **Novelty positioning**: Frame novelty as a precise **delta** against prior blockchain-FL (contribution-aware selection, committee coordination). Do not straw-man prior work as “only passive logging”.
- **Citations**: Every bibliography entry must be cited in text; every citation must resolve to a complete entry with authors, title, venue, year, DOI/URL. Verify venue/year for the two sleeper papers (arXiv IDs given above) and robust-aggregation papers (Krum, Multi-Krum, Bulyan, Trimmed Mean, Median).

---

## Internal consistency rules (strict)

- Any sentence with a number must cite a figure/table and must match it.
- Any mention of PoC must avoid cryptographic proof/consensus language.
- Any novelty claim must name at least two prior works and state the delta precisely.
- Do not describe behavior in figures that is not plotted.
- Do not introduce a symbol without defining it at first use.

End of prompt.
