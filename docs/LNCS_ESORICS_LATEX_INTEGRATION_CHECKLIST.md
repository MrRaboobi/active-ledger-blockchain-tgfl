## LNCS / ESORICS LaTeX Integration + Proofreading Checklist (use after Gemini draft)

This checklist is for the **final stage** when you paste the Gemini-produced draft into the official LNCS template and prepare the submission PDF.

### 1) Template compliance (hard)
- Use the official LNCS `llncs.cls` template **unchanged** (no margin tweaks, no font tweaks).
- Paper length:
  - ≤ **16 pages** excluding bibliography and clearly marked appendices.
  - ≤ **20 pages total**.
- No substantial content moved to appendix. Paper must be intelligible without appendices.
- No colored text/equations; figures must be readable in grayscale.

### 2) Notation + symbol hygiene (reviewer trap)
- Dirichlet concentration: **α** only (e.g., α = 0.5).
- EMA constant: **γ** only (e.g., γ = 0.7). Never reuse α for EMA.
- Define all symbols once. Keep a short “Notation” paragraph near first use.

### 3) Terminology precision (avoid misleading naming)
- Define PoC as a **history-based contribution score** (not cryptographic proof).
- If renaming to “Contribution Score / Validation-based Contribution Score (VCS)”, keep “PoC” acronym consistent once chosen.

### 4) Figures and captions
- Insert figures in this order and ensure every figure is referenced in text:
  - Existing Fig A and Fig B (Session 4 plots you locked).
  - Generated vector figures from `docs/figures_phase3/*.pdf`.
- Use captions and Alt Text drafts from `docs/FIGURE_CAPTIONS_PHASE3.md`.
- Ensure each “number claim” sentence cites the corresponding figure/table.
- Verify grayscale legibility (print preview or convert-to-grayscale check).

### 5) Results consistency (the #1 prior failure mode)
- For every numeric claim:
  - confirm it matches the figure/table value,
  - confirm it matches the Phase 3 experimentation doc,
  - confirm it appears **exactly once** (avoid contradictions).
- Ban vague prose like “stagnates around X–Y%” unless it matches the plotted curve.

### 6) Abstract discipline
- 150–250 words.
- At most 1–2 numeric results.
- Structure: problem → threat → method → key result → security/auditability contribution.

### 7) Related work + novelty positioning
- Ensure novelty is framed as a precise delta:
  - prior blockchain-FL logging and/or coordination,
  - what PoC + trust-gated augmentation adds,
  - what is empirically demonstrated (APB robustness across attacks + augmentation channel closure).
- Remove overclaiming and hype tone.

### 8) References (must be clean)
- Every reference entry must be complete: authors, title, venue, year, DOI/URL.
- No uncited references in bibliography.
- No citations missing from bibliography.
- Verify venue/year for the two sleeper papers and robust aggregation papers.

### 9) Accessibility (Alt Text)
- LNCS typesetting may request Alt Text later. Keep Alt Text drafts in:
  - `docs/FIGURE_CAPTIONS_PHASE3.md`
- During camera-ready/proof stage, verify Alt Text matches figures.

### 10) Final submission artifacts
- Final PDF generated from LaTeX sources.
- All source files: `.tex`, `.bib/.bbl`, figures (prefer `.pdf`), and LNCS class files as required by chairs.

When you bring the LaTeX back, we’ll run this checklist line-by-line and fix anything that violates LNCS/ESORICS constraints.

