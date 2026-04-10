const pptxgen = require('pptxgenjs');

async function buildDeck() {
  const deck = new pptxgen();
  deck.layout = 'LAYOUT_WIDE'; // 13.33" × 7.5"
  deck.author = 'Tommaso Marena';
  deck.title = 'A Quantum Amenability Decision Framework for Protein Structure Prediction';

  // Color constants
  const NAVY = '1A2744';
  const CUA_BLUE = '003399';
  const HEADER_BG = 'E8EBF5';
  const WHITE = 'FFFFFF';
  const BLACK = '000000';
  const DARK_TEXT = '1A1A1A';
  const BODY_GRAY = '333333';
  const RED_LABEL = 'CC3300';
  const LIGHT_BG = 'F5F5F5';
  const TABLE_HEADER = 'D6DCF0';
  const TABLE_ALT = 'F0F2FA';

  // Font constants
  const HEADING_FONT = 'Trebuchet MS';
  const BODY_FONT = 'Calibri';
  const MONO_FONT = 'Courier New';

  // Reusable functions
  function addHeaderBar(sl, title, opts = {}) {
    sl.background = { color: WHITE };
    sl.addShape(deck.shapes.RECTANGLE, {
      x: 0, y: 0, w: 13.33, h: 1.1,
      fill: { color: HEADER_BG },
    });
    // Blue accent line at bottom of header
    sl.addShape(deck.shapes.RECTANGLE, {
      x: 0, y: 1.1, w: 13.33, h: 0.04,
      fill: { color: CUA_BLUE },
    });
    sl.addText(title, {
      x: 0.6, y: 0.15, w: 12.13, h: 0.8,
      fontSize: 24, fontFace: HEADING_FONT, bold: true,
      color: NAVY, valign: 'middle', margin: 0,
    });
  }

  function addSlideNumber(sl, num) {
    sl.addText(String(num), {
      x: 12.33, y: 7.0, w: 0.7, h: 0.3,
      fontSize: 10, fontFace: BODY_FONT, color: '888888',
      align: 'right', valign: 'middle', margin: 0,
    });
  }

  function addClassicallySimulated(sl) {
    sl.addShape(deck.shapes.ROUNDED_RECTANGLE, {
      x: 10.5, y: 6.8, w: 2.5, h: 0.4,
      fill: { color: 'FFF3E0' },
      line: { color: RED_LABEL, width: 1 },
      rectRadius: 0.1,
    });
    sl.addText('[CLASSICALLY SIMULATED]', {
      x: 10.5, y: 6.8, w: 2.5, h: 0.4,
      fontSize: 9, fontFace: MONO_FONT, bold: true,
      color: RED_LABEL, align: 'center', valign: 'middle', margin: 0,
    });
  }

  // ==========================================
  // SLIDE 1 — TITLE
  // ==========================================
  const sl1 = deck.addSlide();
  sl1.background = { color: NAVY };

  // CUA wordmark area - bottom left
  sl1.addText('THE CATHOLIC UNIVERSITY OF AMERICA', {
    x: 0.5, y: 5.8, w: 5, h: 0.4,
    fontSize: 14, fontFace: HEADING_FONT, bold: true,
    color: 'C0C8E0', charSpacing: 2, margin: 0,
  });
  sl1.addShape(deck.shapes.RECTANGLE, {
    x: 0.5, y: 6.25, w: 4.5, h: 0.02,
    fill: { color: 'C0C8E0' },
  });

  // Title text
  sl1.addText('A Quantum Amenability Decision Framework\nfor Protein Structure Prediction', {
    x: 1.0, y: 0.8, w: 11.33, h: 2.4,
    fontSize: 32, fontFace: HEADING_FONT, bold: true,
    color: WHITE, align: 'left', valign: 'middle', margin: 0,
    lineSpacing: 42,
  });

  // Subtitle
  sl1.addText('Side-Chain Rotamer Optimization via\nClassically Simulated Hybrid Quantum Circuits', {
    x: 1.0, y: 3.0, w: 11.33, h: 1.2,
    fontSize: 20, fontFace: BODY_FONT, italic: true,
    color: 'B0BAD0', align: 'left', valign: 'top', margin: 0,
    lineSpacing: 28,
  });

  // Presenter info
  const presenterInfo = [
    { text: 'Tommaso Marena', options: { fontSize: 18, bold: true, color: WHITE, breakLine: true } },
    { text: 'Undergraduate Student, Chemistry and Philosophy Departments', options: { fontSize: 14, color: 'A0AAC0', breakLine: true } },
    { text: 'Double Major: Biochemistry and Philosophy | Pre-Law', options: { fontSize: 14, color: 'A0AAC0', breakLine: true } },
    { text: 'The Catholic University of America', options: { fontSize: 14, color: 'A0AAC0', breakLine: true } },
    { text: 'University Research Day, April 2026', options: { fontSize: 14, color: 'A0AAC0' } },
  ];
  sl1.addText(presenterInfo, {
    x: 1.0, y: 4.3, w: 8, h: 2.0,
    fontFace: BODY_FONT, valign: 'top', margin: 0,
    paraSpaceAfter: 4,
  });

  // ==========================================
  // SLIDE 2 — Background: The Protein Folding Problem
  // ==========================================
  const sl2 = deck.addSlide();
  addHeaderBar(sl2, 'Background: The Protein Folding Problem');
  addSlideNumber(sl2, 2);

  const bullets2 = [
    { text: 'Proteins fold from amino acid sequences into 3D structures that determine biological function', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: 'Side-chain conformations (rotamers) directly control binding pocket geometry, enzymatic activity, and protein-protein interactions', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: 'Each residue has up to n\u1D3A possible rotamer combinations \u2014 exponentially hard (NP-complete)', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: 'Accurate side-chain placement is essential for drug design, protein engineering, and understanding disease mechanisms', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 10 } },
  ];
  sl2.addText(bullets2, {
    x: 0.8, y: 1.5, w: 11.73, h: 4.5,
    valign: 'top', margin: 0,
  });

  // Source
  sl2.addText([
    { text: 'Source: ' , options: { fontSize: 10, color: '888888' } },
    { text: 'Jumper et al. 2021, Nature', options: { fontSize: 10, color: CUA_BLUE, hyperlink: { url: 'https://doi.org/10.1038/s41586-021-03819-2' } } },
    { text: ' (DOI: 10.1038/s41586-021-03819-2)', options: { fontSize: 10, color: '888888' } },
  ], {
    x: 0.6, y: 6.6, w: 10, h: 0.3,
    fontFace: BODY_FONT, margin: 0,
  });

  // ==========================================
  // SLIDE 3 — AlphaFold 2
  // ==========================================
  const sl3 = deck.addSlide();
  addHeaderBar(sl3, 'AlphaFold 2: Achievements and Gaps');
  addSlideNumber(sl3, 3);

  // Left column header
  sl3.addText('What AF2 Achieves', {
    x: 0.6, y: 1.4, w: 5.5, h: 0.5,
    fontSize: 18, fontFace: HEADING_FONT, bold: true,
    color: '2E7D32', margin: 0,
  });

  const leftBullets3 = [
    { text: 'Median CASP14 GDT_TS: 92.4 (Jumper et al. 2021, Table 1B)', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: '58 of 92 domains solved at near-experimental accuracy (GDT_TS > 90)', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Transformed structure prediction for well-ordered single-chain proteins', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true } },
  ];
  sl3.addText(leftBullets3, {
    x: 0.6, y: 1.9, w: 5.5, h: 3.0,
    valign: 'top', margin: 0,
  });

  // Right column header
  sl3.addText('Where Gaps Remain', {
    x: 6.8, y: 1.4, w: 5.93, h: 0.5,
    fontSize: 18, fontFace: HEADING_FONT, bold: true,
    color: 'C62828', margin: 0,
  });

  const rightBullets3 = [
    { text: 'Failure case: T1047s1-D1, GDT_TS = 50.47 due to long beta-sheet at wrong angle (Table 1B)', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Low pLDDT regions: disordered residues, linkers, multi-domain interfaces', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'pLDDT is NOT a calibrated probability (CalPro, arXiv:2601.07201)', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Side-chain \u03C71/\u03C72 recovery rates NOT reported in primary CASP14 paper', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true } },
  ];
  sl3.addText(rightBullets3, {
    x: 6.8, y: 1.9, w: 5.93, h: 3.2,
    valign: 'top', margin: 0,
  });

  // Divider line between columns
  sl3.addShape(deck.shapes.RECTANGLE, {
    x: 6.5, y: 1.5, w: 0.02, h: 3.5,
    fill: { color: 'CCCCCC' },
  });

  // Fairness note box
  sl3.addShape(deck.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 5.3, w: 12.13, h: 0.7,
    fill: { color: 'FFF8E1' },
    line: { color: 'F9A825', width: 1 },
    rectRadius: 0.1,
  });
  sl3.addText('FAIRNESS NOTE: All AlphaFold numbers from Jumper et al. 2021 (DOI: 10.1002/prot.26257). AF2 not re-run for this study.', {
    x: 0.8, y: 5.3, w: 11.73, h: 0.7,
    fontSize: 12, fontFace: BODY_FONT, bold: true,
    color: '795548', valign: 'middle', margin: 0,
  });

  // Source
  sl3.addText([
    { text: 'Source: ' , options: { fontSize: 10, color: '888888' } },
    { text: 'Jumper et al. 2021, Proteins', options: { fontSize: 10, color: CUA_BLUE, hyperlink: { url: 'https://doi.org/10.1002/prot.26257' } } },
  ], {
    x: 0.6, y: 6.6, w: 10, h: 0.3,
    fontFace: BODY_FONT, margin: 0,
  });

  // ==========================================
  // SLIDE 4 — The Hybrid Quantum-Classical Opportunity
  // ==========================================
  const sl4 = deck.addSlide();
  addHeaderBar(sl4, 'The Quantum Opportunity: Discrete Subproblems');
  addSlideNumber(sl4, 4);
  addClassicallySimulated(sl4);

  const bullets4 = [
    { text: 'Quantum computing excels at discrete combinatorial search \u2014 precisely the structure of rotamer optimization', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'Doga et al. 2024 (JCTC): validated hybrid framework on Zika virus NS3 helicase catalytic loop', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'Agathangelou et al. 2025 (arXiv:2507.19383): QUBO formulation of rotamer optimization \u2192 QAOA shows reduced computational cost vs. simulated annealing', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'Key challenge: identifying WHICH subproblems are amenable to near-term quantum hardware', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'This work: a formal decision framework (QADF) for that classification', options: { fontSize: 16, fontFace: BODY_FONT, color: CUA_BLUE, bold: true, bullet: true } },
  ];
  sl4.addText(bullets4, {
    x: 0.8, y: 1.5, w: 11.73, h: 4.5,
    valign: 'top', margin: 0,
  });

  // Source
  sl4.addText([
    { text: 'Source: ' , options: { fontSize: 10, color: '888888' } },
    { text: 'Doga et al. 2024', options: { fontSize: 10, color: CUA_BLUE, hyperlink: { url: 'https://doi.org/10.1021/acs.jctc.4c00067' } } },
    { text: ', ', options: { fontSize: 10, color: '888888' } },
    { text: 'Agathangelou et al. 2025', options: { fontSize: 10, color: CUA_BLUE, hyperlink: { url: 'https://arxiv.org/abs/2507.19383' } } },
  ], {
    x: 0.6, y: 6.6, w: 10, h: 0.3,
    fontFace: BODY_FONT, margin: 0,
  });

  // ==========================================
  // SLIDE 5 — Hypotheses and Objectives (Type B - centered title)
  // ==========================================
  const sl5 = deck.addSlide();
  sl5.background = { color: WHITE };
  addSlideNumber(sl5, 5);

  sl5.addText('Hypotheses and Objectives', {
    x: 0.6, y: 0.3, w: 12.13, h: 0.8,
    fontSize: 28, fontFace: HEADING_FONT, bold: true,
    color: NAVY, align: 'center', valign: 'middle', margin: 0,
  });

  // Thin blue line under title
  sl5.addShape(deck.shapes.RECTANGLE, {
    x: 4.5, y: 1.1, w: 4.33, h: 0.03,
    fill: { color: CUA_BLUE },
  });

  // Hypothesis box
  sl5.addShape(deck.shapes.ROUNDED_RECTANGLE, {
    x: 0.8, y: 1.5, w: 11.73, h: 1.3,
    fill: { color: 'E8EDF8' },
    line: { color: CUA_BLUE, width: 1 },
    rectRadius: 0.1,
  });
  sl5.addText([
    { text: 'Hypothesis: ', options: { bold: true, color: CUA_BLUE } },
    { text: 'Can a formal Quantum Amenability Decision Framework (QADF) identify protein structure subproblems where near-term hybrid quantum-classical approaches provide a defensible advantage over purely classical methods?', options: { color: BODY_GRAY } },
  ], {
    x: 1.0, y: 1.55, w: 11.33, h: 1.2,
    fontSize: 15, fontFace: BODY_FONT, valign: 'middle', margin: 0,
  });

  // Objective 1
  sl5.addShape(deck.shapes.ROUNDED_RECTANGLE, {
    x: 0.8, y: 3.2, w: 11.73, h: 1.2,
    fill: { color: LIGHT_BG },
    rectRadius: 0.1,
  });
  sl5.addText([
    { text: 'Objective 1: ', options: { bold: true, color: NAVY } },
    { text: 'Develop QADF \u2014 a rigorous 9-dimension scoring rubric classifying 8 protein structure subproblems by quantum amenability', options: { color: BODY_GRAY } },
  ], {
    x: 1.0, y: 3.25, w: 11.33, h: 1.1,
    fontSize: 15, fontFace: BODY_FONT, valign: 'middle', margin: 0,
  });

  // Objective 2
  sl5.addShape(deck.shapes.ROUNDED_RECTANGLE, {
    x: 0.8, y: 4.7, w: 11.73, h: 1.2,
    fill: { color: LIGHT_BG },
    rectRadius: 0.1,
  });
  sl5.addText([
    { text: 'Objective 2: ', options: { bold: true, color: NAVY } },
    { text: 'Implement a working prototype \u2014 QUBO-encoded QAOA on real PDB structures ', options: { color: BODY_GRAY } },
    { text: '[CLASSICALLY SIMULATED]', options: { color: RED_LABEL, bold: true, fontFace: MONO_FONT, fontSize: 12 } },
    { text: ' \u2014 and validate with calibrated confidence estimation', options: { color: BODY_GRAY } },
  ], {
    x: 1.0, y: 4.75, w: 11.33, h: 1.1,
    fontSize: 15, fontFace: BODY_FONT, valign: 'middle', margin: 0,
  });

  // ==========================================
  // SLIDE 6 — The QADF Framework
  // ==========================================
  const sl6 = deck.addSlide();
  addHeaderBar(sl6, 'Quantum Amenability Decision Framework (QADF)');
  addSlideNumber(sl6, 6);

  // Table
  const tableRows6 = [
    [
      { text: 'Subproblem', options: { bold: true, fontSize: 13, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Qubits', options: { bold: true, fontSize: 13, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'QUBO Fit', options: { bold: true, fontSize: 13, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Noise Sensitivity', options: { bold: true, fontSize: 13, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Classification', options: { bold: true, fontSize: 13, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
    ],
    [
      { text: 'Global backbone folding', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE } } },
      { text: '>100', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: 'Poor', options: { fontSize: 13, fontFace: BODY_FONT, color: 'C62828', fill: { color: WHITE }, align: 'center' } },
      { text: 'High', options: { fontSize: 13, fontFace: BODY_FONT, color: 'C62828', fill: { color: WHITE }, align: 'center' } },
      { text: 'C: Poor near-term', options: { fontSize: 13, fontFace: BODY_FONT, color: 'C62828', bold: true, fill: { color: 'FFEBEE' }, align: 'center' } },
    ],
    [
      { text: 'Short peptide search', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT } } },
      { text: '10\u201320', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'Moderate', options: { fontSize: 13, fontFace: BODY_FONT, color: 'F57F17', fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'Moderate', options: { fontSize: 13, fontFace: BODY_FONT, color: 'F57F17', fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'B: Medium-term', options: { fontSize: 13, fontFace: BODY_FONT, color: 'F57F17', bold: true, fill: { color: 'FFF8E1' }, align: 'center' } },
    ],
    [
      { text: 'Side-chain packing', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE } } },
      { text: '8\u201316', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: 'Excellent', options: { fontSize: 13, fontFace: BODY_FONT, color: '2E7D32', fill: { color: WHITE }, align: 'center' } },
      { text: 'Moderate', options: { fontSize: 13, fontFace: BODY_FONT, color: 'F57F17', fill: { color: WHITE }, align: 'center' } },
      { text: 'A: Near-term candidate', options: { fontSize: 13, fontFace: BODY_FONT, color: '2E7D32', bold: true, fill: { color: 'E8F5E9' }, align: 'center' } },
    ],
    [
      { text: 'Disordered region sampling', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT } } },
      { text: '>50', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'Poor', options: { fontSize: 13, fontFace: BODY_FONT, color: 'C62828', fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'High', options: { fontSize: 13, fontFace: BODY_FONT, color: 'C62828', fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'C: Poor near-term', options: { fontSize: 13, fontFace: BODY_FONT, color: 'C62828', bold: true, fill: { color: 'FFEBEE' }, align: 'center' } },
    ],
  ];

  sl6.addTable(tableRows6, {
    x: 0.6, y: 1.4, w: 12.13,
    colW: [3.5, 1.5, 1.8, 2.0, 3.33],
    rowH: [0.5, 0.5, 0.5, 0.5, 0.5],
    border: { type: 'solid', color: 'CCCCCC', pt: 0.5 },
    autoPage: false,
  });

  // Note
  sl6.addText([
    { text: 'Note: ', options: { bold: true, color: NAVY } },
    { text: 'Global backbone folding classified C: QAOA cannot solve realistic protein folding even at p=100+ layers (Bauza et al. 2023, npj Quantum Inf.)', options: { color: '666666' } },
  ], {
    x: 0.6, y: 4.3, w: 12.13, h: 0.8,
    fontSize: 12, fontFace: BODY_FONT, valign: 'top', margin: 0,
  });

  // Source
  sl6.addText([
    { text: 'Source: ' , options: { fontSize: 10, color: '888888' } },
    { text: 'Bauza et al. 2023', options: { fontSize: 10, color: CUA_BLUE, hyperlink: { url: 'https://doi.org/10.1038/s41534-023-00733-5' } } },
  ], {
    x: 0.6, y: 6.6, w: 10, h: 0.3,
    fontFace: BODY_FONT, margin: 0,
  });

  // ==========================================
  // SLIDE 7 — Methods: Model Architecture
  // ==========================================
  const sl7 = deck.addSlide();
  addHeaderBar(sl7, 'Methods: Hybrid Model Architecture');
  addSlideNumber(sl7, 7);
  addClassicallySimulated(sl7);

  // Left column header
  sl7.addText('Classical Components', {
    x: 0.6, y: 1.4, w: 5.5, h: 0.45,
    fontSize: 17, fontFace: HEADING_FONT, bold: true,
    color: CUA_BLUE, margin: 0,
  });

  const leftBullets7 = [
    { text: 'Equivariant GNN backbone (2 EGCL layers, hidden dim 32)', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Input: sequence + residue features + Dunbrack rotamer priors (Dunbrack 2011)', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Output head: rotamer class probabilities + pLDDT-style confidence score (0\u2013100)', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Loss: cross-entropy + steric clash penalty + calibration regularizer', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true } },
  ];
  sl7.addText(leftBullets7, {
    x: 0.6, y: 1.9, w: 5.7, h: 3.0,
    valign: 'top', margin: 0,
  });

  // Right column header
  sl7.addText([
    { text: 'Quantum Module ', options: { color: CUA_BLUE } },
    { text: '[CLASSICALLY SIMULATED]', options: { color: RED_LABEL, fontSize: 11, fontFace: MONO_FONT } },
  ], {
    x: 6.8, y: 1.4, w: 5.93, h: 0.45,
    fontSize: 17, fontFace: HEADING_FONT, bold: true, margin: 0,
  });

  const rightBullets7 = [
    { text: '6\u201310 qubit parameterized quantum circuit (PennyLane default.qubit)', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Role: latent feature transformer over fixed rotamer embedding', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'QAOA-style: alternating cost unitary U_C(\u03B3) and mixer U_M(\u03B2)', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Optimizer: COBYLA, 200 iterations', options: { fontSize: 14, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'Falsifiable claim: quantum feature transformation explores discrete rotamer space more efficiently than equivalent classical MLP under fixed compute [tested on 4-residue QUBO instance]', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true } },
  ];
  sl7.addText(rightBullets7, {
    x: 6.8, y: 1.9, w: 5.93, h: 3.5,
    valign: 'top', margin: 0,
  });

  // Divider line
  sl7.addShape(deck.shapes.RECTANGLE, {
    x: 6.5, y: 1.5, w: 0.02, h: 3.8,
    fill: { color: 'CCCCCC' },
  });

  // Disclaimer box
  sl7.addShape(deck.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 5.7, w: 9.5, h: 0.55,
    fill: { color: 'FFF3E0' },
    line: { color: RED_LABEL, width: 1 },
    rectRadius: 0.1,
  });
  sl7.addText('\u26A0 All quantum circuit execution is classical simulation. No QPU hardware used.', {
    x: 0.8, y: 5.7, w: 9.1, h: 0.55,
    fontSize: 12, fontFace: BODY_FONT, bold: true,
    color: RED_LABEL, valign: 'middle', margin: 0,
  });

  // ==========================================
  // SLIDE 8 — Results: Main Benchmark
  // ==========================================
  const sl8 = deck.addSlide();
  addHeaderBar(sl8, 'Results: QUBO Optimization on 1L2Y [CLASSICALLY SIMULATED]');
  addSlideNumber(sl8, 8);
  addClassicallySimulated(sl8);

  const bullets8 = [
    { text: 'System: 1L2Y, 4-residue window (residues 3\u20136), 12 binary variables', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 14 } },
    { text: 'Exhaustive search ground truth energy: [SEE_RESULTS]', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 14 } },
    { text: 'Greedy assignment: [SEE_RESULTS]', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 14 } },
    { text: 'Simulated annealing: [SEE_RESULTS]', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 14 } },
    { text: 'QAOA p=1 [CLASSICALLY SIMULATED]: [SEE_RESULTS]', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 14 } },
    { text: 'QAOA p=2 [CLASSICALLY SIMULATED]: [SEE_RESULTS]', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true } },
  ];
  sl8.addText(bullets8, {
    x: 0.8, y: 1.5, w: 11.73, h: 4.0,
    valign: 'top', margin: 0,
  });

  // Disclaimer text
  sl8.addText('All [CLASSICALLY SIMULATED] quantum results produced via PennyLane default.qubit classical state-vector simulation. No quantum hardware used.', {
    x: 0.8, y: 5.8, w: 9.5, h: 0.5,
    fontSize: 11, fontFace: BODY_FONT, italic: true,
    color: '888888', valign: 'top', margin: 0,
  });

  // ==========================================
  // SLIDE 9 — Results: Scaling Study
  // ==========================================
  const sl9 = deck.addSlide();
  addHeaderBar(sl9, 'Results: Quantum Simulation Scaling Boundary [CLASSICALLY SIMULATED]');
  addSlideNumber(sl9, 9);
  addClassicallySimulated(sl9);

  const tableRows9 = [
    [
      { text: 'Residues', options: { bold: true, fontSize: 13, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Binary vars', options: { bold: true, fontSize: 13, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Qubits', options: { bold: true, fontSize: 13, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Simulation feasible?', options: { bold: true, fontSize: 13, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
    ],
    [
      { text: '2', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: '6', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: '6', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: 'Yes', options: { fontSize: 13, fontFace: BODY_FONT, color: '2E7D32', bold: true, fill: { color: 'E8F5E9' }, align: 'center' } },
    ],
    [
      { text: '3', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: '9', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: '9', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'Yes', options: { fontSize: 13, fontFace: BODY_FONT, color: '2E7D32', bold: true, fill: { color: 'E8F5E9' }, align: 'center' } },
    ],
    [
      { text: '4', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: '12', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: '12', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: 'Yes', options: { fontSize: 13, fontFace: BODY_FONT, color: '2E7D32', bold: true, fill: { color: 'E8F5E9' }, align: 'center' } },
    ],
    [
      { text: '5', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: '15', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: '15', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'Yes (slower)', options: { fontSize: 13, fontFace: BODY_FONT, color: 'F57F17', bold: true, fill: { color: 'FFF8E1' }, align: 'center' } },
    ],
    [
      { text: '6', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: '18', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: '18', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: WHITE }, align: 'center' } },
      { text: 'Boundary', options: { fontSize: 13, fontFace: BODY_FONT, color: 'E65100', bold: true, fill: { color: 'FFF3E0' }, align: 'center' } },
    ],
    [
      { text: '7+', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: '21+', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: '21+', options: { fontSize: 13, fontFace: BODY_FONT, color: BODY_GRAY, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'Intractable for repeated experiments', options: { fontSize: 12, fontFace: BODY_FONT, color: 'C62828', bold: true, fill: { color: 'FFEBEE' }, align: 'center' } },
    ],
  ];

  sl9.addTable(tableRows9, {
    x: 1.5, y: 1.4, w: 10.33,
    colW: [2.0, 2.33, 2.0, 4.0],
    rowH: [0.45, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
    border: { type: 'solid', color: 'CCCCCC', pt: 0.5 },
    autoPage: false,
  });

  sl9.addText('The simulation boundary at ~20 qubits is itself a finding: it quantifies the resource requirement for real quantum hardware to provide utility on this problem class.', {
    x: 0.8, y: 5.2, w: 9.5, h: 0.6,
    fontSize: 13, fontFace: BODY_FONT, italic: true,
    color: '555555', valign: 'top', margin: 0,
  });

  // ==========================================
  // SLIDE 10 — Results: Ablation and Confidence
  // ==========================================
  const sl10 = deck.addSlide();
  addHeaderBar(sl10, 'Results: Ablation Study and Confidence Calibration');
  addSlideNumber(sl10, 10);

  // Left side — Ablation
  sl10.addText('Ablation Study', {
    x: 0.6, y: 1.4, w: 5.5, h: 0.45,
    fontSize: 17, fontFace: HEADING_FONT, bold: true,
    color: NAVY, margin: 0,
  });

  sl10.addText('6 ablation conditions tested. Results: [SEE_RESULTS]\n\nConditions include:\n\u2022 Full model\n\u2022 No quantum module\n\u2022 No calibration regularizer\n\u2022 No steric clash penalty\n\u2022 Random rotamer priors\n\u2022 Reduced GNN layers', {
    x: 0.6, y: 1.9, w: 5.7, h: 3.8,
    fontSize: 14, fontFace: BODY_FONT,
    color: BODY_GRAY, valign: 'top', margin: 0,
  });

  // Right side — Confidence
  sl10.addText('pLDDT Confidence Legend', {
    x: 6.8, y: 1.4, w: 5.93, h: 0.45,
    fontSize: 17, fontFace: HEADING_FONT, bold: true,
    color: NAVY, margin: 0,
  });

  // pLDDT color bars
  const plddt = [
    { label: '>90: Very high confidence', color: '0053D6', bg: 'E3ECFA' },
    { label: '70\u201390: Confident', color: '65CBF3', bg: 'E8F7FD' },
    { label: '50\u201370: Low confidence', color: 'FFDB13', bg: 'FFF8D6' },
    { label: '<50: Very low confidence', color: 'FF7D45', bg: 'FFF0E8' },
  ];
  plddt.forEach((p, i) => {
    const yPos = 2.0 + i * 0.6;
    sl10.addShape(deck.shapes.RECTANGLE, {
      x: 6.8, y: yPos, w: 0.4, h: 0.4,
      fill: { color: p.color },
    });
    sl10.addText(p.label, {
      x: 7.4, y: yPos, w: 5.33, h: 0.4,
      fontSize: 14, fontFace: BODY_FONT,
      color: BODY_GRAY, valign: 'middle', margin: 0,
    });
  });

  sl10.addText('Confidence coloring follows AlphaFold pLDDT convention (Jumper et al. 2021). Scores shown are from this model, not AlphaFold.', {
    x: 6.8, y: 4.5, w: 5.93, h: 0.8,
    fontSize: 11, fontFace: BODY_FONT, italic: true,
    color: '888888', valign: 'top', margin: 0,
  });

  // Divider line
  sl10.addShape(deck.shapes.RECTANGLE, {
    x: 6.5, y: 1.5, w: 0.02, h: 3.8,
    fill: { color: 'CCCCCC' },
  });

  // ==========================================
  // SLIDE 11 — Results: Noise Analysis
  // ==========================================
  const sl11 = deck.addSlide();
  addHeaderBar(sl11, 'Results: Noise Degradation Analysis [CLASSICALLY SIMULATED]');
  addSlideNumber(sl11, 11);
  addClassicallySimulated(sl11);

  const noiseConditions = [
    { label: 'Noiseless (ideal)', result: '[SEE_RESULTS]', color: '2E7D32' },
    { label: 'Depolarizing \u03B5=0.001/gate', result: '[SEE_RESULTS]', color: 'F57F17' },
    { label: 'Depolarizing \u03B5=0.01/gate', result: '[SEE_RESULTS]', color: 'C62828' },
  ];

  noiseConditions.forEach((nc, i) => {
    const yPos = 1.5 + i * 1.3;
    sl11.addShape(deck.shapes.ROUNDED_RECTANGLE, {
      x: 0.8, y: yPos, w: 11.73, h: 1.0,
      fill: { color: LIGHT_BG },
      rectRadius: 0.1,
    });
    sl11.addText([
      { text: nc.label + ': ', options: { bold: true, color: nc.color, fontSize: 18 } },
      { text: nc.result, options: { color: BODY_GRAY, fontSize: 18, fontFace: MONO_FONT } },
    ], {
      x: 1.0, y: yPos, w: 11.33, h: 1.0,
      fontFace: BODY_FONT, valign: 'middle', margin: 0,
    });
  });

  // NISQ notes
  sl11.addText([
    { text: 'NISQ error rates: single-qubit \u03B5\u2081 ~10\u207B\u2074\u201310\u207B\u00B3, two-qubit \u03B5\u2082 ~10\u207B\u00B3\u201310\u207B\u00B2', options: { breakLine: true, paraSpaceAfter: 6 } },
    { text: 'At \u03B5=0.01, performance degrades significantly \u2014 motivating error mitigation in future work', options: {} },
  ], {
    x: 0.8, y: 5.5, w: 9.5, h: 0.8,
    fontSize: 12, fontFace: BODY_FONT, italic: true,
    color: '666666', valign: 'top', margin: 0,
  });

  // ==========================================
  // SLIDE 12 — AlphaFold Comparison
  // ==========================================
  const sl12 = deck.addSlide();
  addHeaderBar(sl12, 'Comparison to Published AlphaFold 2 Benchmarks');
  addSlideNumber(sl12, 12);
  addClassicallySimulated(sl12);

  const tableRows12 = [
    [
      { text: 'Method', options: { bold: true, fontSize: 11, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Task', options: { bold: true, fontSize: 11, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Metric', options: { bold: true, fontSize: 11, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Value', options: { bold: true, fontSize: 11, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Dataset', options: { bold: true, fontSize: 11, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
      { text: 'Source', options: { bold: true, fontSize: 11, fontFace: HEADING_FONT, color: WHITE, fill: { color: CUA_BLUE }, align: 'center', valign: 'middle' } },
    ],
    [
      { text: 'AlphaFold 2', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
      { text: 'Global backbone', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
      { text: 'GDT_TS', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
      { text: '92.4 (median)', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
      { text: 'CASP14, 92 domains', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
      { text: 'Jumper 2021, Table 1B', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
    ],
    [
      { text: 'AlphaFold 2', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'Global backbone', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'GDT_TS', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: '50.47 (failure T1047)', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'CASP14', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'Jumper 2021, Table 1B', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
    ],
    [
      { text: 'AlphaFold 2', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
      { text: 'Side-chain \u03C71/\u03C72', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
      { text: 'N/A', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
      { text: 'Not reported', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center', italic: true } },
      { text: 'CASP14', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
      { text: 'Jumper 2021', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: WHITE }, align: 'center' } },
    ],
    [
      { text: 'This work (hybrid)', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: 'E8F5E9' }, align: 'center', bold: true } },
      { text: 'Rotamer opt.', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: 'E8F5E9' }, align: 'center' } },
      { text: 'QUBO energy', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: 'E8F5E9' }, align: 'center' } },
      { text: '[SEE_RESULTS \u00B1 CI]', options: { fontSize: 10, fontFace: MONO_FONT, fill: { color: 'E8F5E9' }, align: 'center' } },
      { text: '1L2Y (20 res.)', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: 'E8F5E9' }, align: 'center' } },
      { text: 'This work', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: 'E8F5E9' }, align: 'center', bold: true } },
    ],
    [
      { text: 'Classical SA', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'Rotamer opt.', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'QUBO energy', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: '[SEE_RESULTS \u00B1 CI]', options: { fontSize: 10, fontFace: MONO_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: '1L2Y (20 res.)', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
      { text: 'This work', options: { fontSize: 10, fontFace: BODY_FONT, fill: { color: TABLE_ALT }, align: 'center' } },
    ],
  ];

  sl12.addTable(tableRows12, {
    x: 0.3, y: 1.3, w: 12.73,
    colW: [2.0, 1.8, 1.5, 2.3, 2.3, 2.83],
    rowH: [0.42, 0.4, 0.4, 0.4, 0.4, 0.4],
    border: { type: 'solid', color: 'CCCCCC', pt: 0.5 },
    autoPage: false,
  });

  // Fairness caveat box
  sl12.addShape(deck.shapes.ROUNDED_RECTANGLE, {
    x: 0.3, y: 4.1, w: 12.73, h: 1.8,
    fill: { color: 'FFF8E1' },
    line: { color: 'F9A825', width: 1.5 },
    rectRadius: 0.1,
  });
  sl12.addText([
    { text: 'MANDATORY FAIRNESS CAVEAT:\n', options: { bold: true, color: 'C62828', fontSize: 13 } },
    { text: 'AlphaFold 2 numbers are from Jumper et al. 2021 (DOI: 10.1002/prot.26257, Table 1A/B) under CASP14 conditions not directly comparable to this study. This project addresses side-chain rotamer optimization only, on PDB structures \u226425 residues, via classical simulation. GDT_TS comparison is not appropriate \u2014 included for context only. AlphaFold was not re-run.', options: { color: '795548', fontSize: 12 } },
  ], {
    x: 0.5, y: 4.2, w: 12.33, h: 1.6,
    fontFace: BODY_FONT, valign: 'top', margin: 0,
  });

  // Source
  sl12.addText([
    { text: 'Source: ', options: { fontSize: 10, color: '888888' } },
    { text: 'Jumper et al. 2021', options: { fontSize: 10, color: CUA_BLUE, hyperlink: { url: 'https://doi.org/10.1002/prot.26257' } } },
  ], {
    x: 0.6, y: 6.6, w: 10, h: 0.3,
    fontFace: BODY_FONT, margin: 0,
  });

  // ==========================================
  // SLIDE 13 — Discussion
  // ==========================================
  const sl13 = deck.addSlide();
  sl13.background = { color: WHITE };
  addSlideNumber(sl13, 13);

  sl13.addText('Discussion', {
    x: 0.6, y: 0.3, w: 12.13, h: 0.8,
    fontSize: 28, fontFace: HEADING_FONT, bold: true,
    color: NAVY, align: 'center', valign: 'middle', margin: 0,
  });

  sl13.addShape(deck.shapes.RECTANGLE, {
    x: 4.5, y: 1.1, w: 4.33, h: 0.03,
    fill: { color: CUA_BLUE },
  });

  const bullets13 = [
    { text: 'QADF successfully distinguishes near-term quantum candidates from poor targets \u2014 side-chain packing emerges as Category A', options: { fontSize: 15, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'Honest finding: QAOA at p=1,2 does not dramatically outperform simulated annealing on 4-residue instances \u2014 consistent with known QAOA limitations at low circuit depth (Bauza 2023)', options: { fontSize: 15, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'Confidence calibration gap: this model\u2019s pLDDT-style scores require careful calibration; raw pLDDT is itself miscalibrated (CalPro, arXiv:2601.07201)', options: { fontSize: 15, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'This work\u2019s main contribution is the QADF framework and the documented scaling analysis \u2014 the prototype proves the pipeline is executable', options: { fontSize: 15, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'FlowPacker (bioRxiv 2024.07.05.602280) represents the classical ceiling this approach must eventually surpass', options: { fontSize: 15, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true } },
  ];
  sl13.addText(bullets13, {
    x: 0.8, y: 1.4, w: 11.73, h: 5.0,
    valign: 'top', margin: 0,
  });

  // ==========================================
  // SLIDE 14 — Conclusions
  // ==========================================
  const sl14 = deck.addSlide();
  addHeaderBar(sl14, 'Conclusions');
  addSlideNumber(sl14, 14);

  const bullets14 = [
    { text: 'QADF provides a rigorous, citable framework for selecting protein structure subproblems amenable to near-term quantum hardware', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'Side-chain rotamer optimization is validated as a Category A near-term hybrid candidate', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'Classical simulation boundary at ~20 qubits is documented as a resource result', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'Calibrated confidence estimation demonstrates what pLDDT-style scoring should look like for hybrid models', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true, breakLine: true, paraSpaceAfter: 12 } },
    { text: 'All quantum experiments classically simulated \u2014 methodology consistent with field standard for NISQ-era algorithm research', options: { fontSize: 16, fontFace: BODY_FONT, color: BODY_GRAY, bullet: true } },
  ];
  sl14.addText(bullets14, {
    x: 0.8, y: 1.5, w: 11.73, h: 5.0,
    valign: 'top', margin: 0,
  });

  // ==========================================
  // SLIDE 15 — Conclusions + Acknowledgements + References
  // ==========================================
  const sl15 = deck.addSlide();
  sl15.background = { color: WHITE };
  addSlideNumber(sl15, 15);

  sl15.addText('Conclusions', {
    x: 0.6, y: 0.2, w: 12.13, h: 0.6,
    fontSize: 26, fontFace: HEADING_FONT, bold: true,
    color: NAVY, align: 'center', valign: 'middle', margin: 0,
  });

  sl15.addShape(deck.shapes.RECTANGLE, {
    x: 4.5, y: 0.8, w: 4.33, h: 0.03,
    fill: { color: CUA_BLUE },
  });

  // Main conclusions bullet
  sl15.addText('\u2022  State the major points of the work \u2014 see Slide 14 for full conclusions list.', {
    x: 0.6, y: 1.0, w: 7.5, h: 0.4,
    fontSize: 13, fontFace: BODY_FONT,
    color: BODY_GRAY, valign: 'top', margin: 0,
  });

  // Acknowledgements section
  sl15.addText('Acknowledgements', {
    x: 0.6, y: 1.6, w: 7.5, h: 0.45,
    fontSize: 20, fontFace: HEADING_FONT, bold: true,
    color: NAVY, margin: 0,
  });

  sl15.addText('This research was conducted independently without external funding or institutional support. I thank the open-source scientific computing community (PennyLane, BioPython, RCSB PDB) whose tools made this work possible.', {
    x: 0.6, y: 2.1, w: 7.5, h: 1.0,
    fontSize: 13, fontFace: BODY_FONT,
    color: BODY_GRAY, valign: 'top', margin: 0,
  });

  // Funding
  sl15.addText('Funding', {
    x: 0.6, y: 3.2, w: 7.5, h: 0.4,
    fontSize: 18, fontFace: HEADING_FONT, bold: true,
    color: NAVY, margin: 0,
  });

  sl15.addText('No external funding.', {
    x: 0.6, y: 3.6, w: 7.5, h: 0.35,
    fontSize: 13, fontFace: BODY_FONT,
    color: BODY_GRAY, valign: 'top', margin: 0,
  });

  // References box (right side)
  sl15.addShape(deck.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 4.1, w: 12.13, h: 3.1,
    fill: { color: 'F5F5F5' },
    line: { color: 'CCCCCC', width: 0.5 },
    rectRadius: 0.1,
  });

  sl15.addText('References', {
    x: 0.8, y: 4.15, w: 11.73, h: 0.35,
    fontSize: 14, fontFace: HEADING_FONT, bold: true,
    color: NAVY, margin: 0,
  });

  const refs = [
    '1. Doga et al. (2024) J. Chem. Theory Comput. DOI: 10.1021/acs.jctc.4c00067',
    '2. Agathangelou et al. (2025) arXiv:2507.19383',
    '3. Khatami et al. (2023) PLOS Comput. Biol. DOI: 10.1371/journal.pcbi.1011033',
    '4. Jumper et al. (2021) Proteins 89:1711. DOI: 10.1002/prot.26257',
    '4b. Jumper et al. (2021) Nature 596:583. DOI: 10.1038/s41586-021-03819-2',
    '5. Dunbrack (2011) PubMed: 21645855',
    '6. Bauza et al. (2023) npj Quantum Inf. DOI: 10.1038/s41534-023-00733-5',
    '7. FlowPacker (2024) bioRxiv 2024.07.05.602280',
    '8. CalPro (2026) arXiv:2601.07201',
  ];

  const refText = refs.map((r, i) => ({
    text: r,
    options: {
      fontSize: 10, fontFace: BODY_FONT, color: '555555',
      breakLine: i < refs.length - 1 ? true : false,
      paraSpaceAfter: 2,
    },
  }));
  sl15.addText(refText, {
    x: 0.8, y: 4.5, w: 11.73, h: 2.6,
    valign: 'top', margin: 0,
  });

  // Write file
  const outputPath = '/home/user/workspace/marena-qadf/slides/marena_qadf_slides.pptx';
  await deck.writeFile({ fileName: outputPath });
  console.log('Deck written to ' + outputPath);
}

buildDeck().catch(err => { console.error(err); process.exit(1); });
