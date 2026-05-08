# Experiment Analysis -- 2-Mode Evaluation

Sources: ['outputs\\eval_vec.csv', 'outputs\\eval_graph.csv', 'outputs\\eval_iter.csv']
N success rows: 84 total  |  84 in main comparison  |  Modes: ['GRAPH_ITERATIVE', 'GRAPH_RETRIEVAL', 'VECTOR_RETRIEVAL']  |  Question types: ['CROSSREF', 'MULTIHOP']


---

## Table 1 -- Overall per-mode scores

| Mode             | N  | BERTScore F1 mean (sd) | ROUGE-L mean | Recall@3 mean | Latency ms mean |
| ---------------- | -- | ---------------------- | ------------ | ------------- | --------------- |
| VECTOR_RETRIEVAL | 28 | 0.623 (0.127)          | 0.278        | 0.619         | 5362            |
| GRAPH_RETRIEVAL  | 28 | 0.650 (0.117)          | 0.324        | 0.577         | 6475            |
| GRAPH_ITERATIVE  | 28 | 0.660 (0.100)          | 0.329        | 0.577         | 136260          |

---

## Table 2 -- Per-mode x question-type scores


### BERTScore F1

| Type        | VECTOR_RETRIEVAL | GRAPH_RETRIEVAL | GRAPH_ITERATIVE | N (VEC/GR) |
| ----------- | ---------------- | --------------- | --------------- | ---------- |
| CROSSREF    | 0.616            | 0.639           | 0.662           | N=14/14    |
| MULTIHOP    | 0.630            | 0.661           | 0.658           | N=14/14    |
| **Overall** | 0.623            | 0.650           | 0.660           |            |

### ROUGE-L

| Type        | VECTOR_RETRIEVAL | GRAPH_RETRIEVAL | GRAPH_ITERATIVE | N (VEC/GR) |
| ----------- | ---------------- | --------------- | --------------- | ---------- |
| CROSSREF    | 0.295            | 0.336           | 0.345           | N=14/14    |
| MULTIHOP    | 0.261            | 0.312           | 0.313           | N=14/14    |
| **Overall** | 0.278            | 0.324           | 0.329           |            |

### Recall@3

| Type        | VECTOR_RETRIEVAL | GRAPH_RETRIEVAL | GRAPH_ITERATIVE | N (VEC/GR) |
| ----------- | ---------------- | --------------- | --------------- | ---------- |
| CROSSREF    | 0.643            | 0.536           | 0.536           | N=14/14    |
| MULTIHOP    | 0.595            | 0.619           | 0.619           | N=14/14    |
| **Overall** | 0.619            | 0.577           | 0.577           |            |

### Complete@3

| Type        | VECTOR_RETRIEVAL | GRAPH_RETRIEVAL | GRAPH_ITERATIVE | N (VEC/GR) |
| ----------- | ---------------- | --------------- | --------------- | ---------- |
| CROSSREF    | 0.286            | 0.143           | 0.143           | N=14/14    |
| MULTIHOP    | 0.214            | 0.214           | 0.214           | N=14/14    |
| **Overall** | 0.250            | 0.179           | 0.179           |            |

---

## Table 3 -- Paired Wilcoxon signed-rank test (proposed > VECTOR baseline)

alpha = 0.05, one-tailed.


### GRAPH_RETRIEVAL vs VECTOR_RETRIEVAL

| Metric       | N pairs | VECTOR mean | Challenger mean | Delta  | Wilcoxon p | t-test p | Sig (W) | Sig (t) |
| ------------ | ------- | ----------- | --------------- | ------ | ---------- | -------- | ------- | ------- |
| BERTScore F1 | 28      | 0.623       | 0.650           | 0.027  | 0.046      | 0.057    | *       | ns      |
| ROUGE-L      | 28      | 0.278       | 0.324           | 0.046  | 0.013      | 0.016    | *       | *       |
| Recall@3     | 28      | 0.619       | 0.577           | -0.042 | 0.828      | 0.761    | ns      | ns      |
| Complete@3   | 28      | 0.250       | 0.179           | -0.071 | 0.760      | 0.755    | ns      | ns      |

### GRAPH_ITERATIVE vs VECTOR_RETRIEVAL

| Metric       | N pairs | VECTOR mean | Challenger mean | Delta  | Wilcoxon p | t-test p | Sig (W) | Sig (t) |
| ------------ | ------- | ----------- | --------------- | ------ | ---------- | -------- | ------- | ------- |
| BERTScore F1 | 28      | 0.623       | 0.660           | 0.036  | 0.078      | 0.068    | ns      | ns      |
| ROUGE-L      | 28      | 0.278       | 0.329           | 0.051  | 0.027      | 0.027    | *       | *       |
| Recall@3     | 28      | 0.619       | 0.577           | -0.042 | 0.828      | 0.761    | ns      | ns      |
| Complete@3   | 28      | 0.250       | 0.179           | -0.071 | 0.760      | 0.755    | ns      | ns      |

_*p<0.05  **p<0.01  ***p<0.001  ns=not significant_

---

## Table 4 -- Wilcoxon stratified by question type (BERTScore F1)

| Challenger      | Type     | N pairs | VECTOR mean | Challenger mean | Delta | Wilcoxon p | t-test p | Sig (W) | Sig (t) |
| --------------- | -------- | ------- | ----------- | --------------- | ----- | ---------- | -------- | ------- | ------- |
| GRAPH_RETRIEVAL | CROSSREF | 14      | 0.616       | 0.639           | 0.023 | 0.091      | 0.156    | ns      | ns      |
| GRAPH_RETRIEVAL | MULTIHOP | 14      | 0.630       | 0.661           | 0.031 | 0.121      | 0.123    | ns      | ns      |
| GRAPH_ITERATIVE | CROSSREF | 14      | 0.616       | 0.662           | 0.046 | 0.232      | 0.126    | ns      | ns      |
| GRAPH_ITERATIVE | MULTIHOP | 14      | 0.630       | 0.658           | 0.027 | 0.086      | 0.186    | ns      | ns      |

_*p<0.05  **p<0.01  ***p<0.001  ns=not significant_

---

## Table 5 -- Retrieval metrics (Hit@3, Recall@3)

| Mode                      | N  | hit@3 | recall@3 | precision@3 | complete@3 |
| ------------------------- | -- | ----- | -------- | ----------- | ---------- |
| VECTOR_RETRIEVAL          | 28 | 0.964 | 0.619    | 0.440       | 0.250      |
| GRAPH_RETRIEVAL           | 28 | 0.964 | 0.577    | 0.417       | 0.179      |
| GRAPH_ITERATIVE           | 28 | 0.964 | 0.577    | 0.417       | 0.179      |
| VECTOR_RETRIEVAL/CROSSREF | 14 | 1.000 | 0.643    | 0.428       | 0.286      |
| GRAPH_RETRIEVAL/CROSSREF  | 14 | 0.929 | 0.536    | 0.357       | 0.143      |
| GRAPH_ITERATIVE/CROSSREF  | 14 | 0.929 | 0.536    | 0.357       | 0.143      |
| VECTOR_RETRIEVAL/MULTIHOP | 14 | 0.929 | 0.595    | 0.452       | 0.214      |
| GRAPH_RETRIEVAL/MULTIHOP  | 14 | 1.000 | 0.619    | 0.476       | 0.214      |
| GRAPH_ITERATIVE/MULTIHOP  | 14 | 1.000 | 0.619    | 0.476       | 0.214      |

---

## Table 7 -- Narrative summary for Chapter 6

- Best mode (BERTScore F1): **GRAPH_ITERATIVE** (0.660)
- Worst mode: **VECTOR_RETRIEVAL** (0.623)
- Delta GRAPH_RETRIEVAL vs VECTOR (BERTScore F1): 0.027
- Delta GRAPH_ITERATIVE vs VECTOR (BERTScore F1): 0.036
- GRAPH_RETRIEVAL overall (N=28): Wilcoxon p=0.046 * -- SIGNIFICANT
- GRAPH_RETRIEVAL overall (N=28):  t-test  p=0.057 ns -- not significant
  - GRAPH_RETRIEVAL/CROSSREF (N=14): delta=0.023, Wilcoxon p=0.091 ns, t-test p=0.156 ns
  - GRAPH_RETRIEVAL/MULTIHOP (N=14): delta=0.031, Wilcoxon p=0.121 ns, t-test p=0.123 ns
- GRAPH_ITERATIVE overall (N=28): Wilcoxon p=0.078 ns -- not significant
- GRAPH_ITERATIVE overall (N=28):  t-test  p=0.068 ns -- not significant
  - GRAPH_ITERATIVE/CROSSREF (N=14): delta=0.046, Wilcoxon p=0.232 ns, t-test p=0.126 ns
  - GRAPH_ITERATIVE/MULTIHOP (N=14): delta=0.027, Wilcoxon p=0.086 ns, t-test p=0.186 ns