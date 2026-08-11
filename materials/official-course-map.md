# learn-unsupervised-with-phoebe - official course map

Built 2026-08-11 from verified official sources (course-taking loop paused - built direct).
Bucket: ds, difficulty 3, builder track only, 10 sessions (u1-u10) - the no-labels sibling of
learn-classification-regression's algorithm zoo.
Running case: **Mango Lane** (fashion app, ecom-bucket canon: AOV $68, monthly GMV $660,960).
Marketing's brief: "12,000 customers, no labels - give us segments we can actually campaign to."
The lab works on a deterministic 300-customer sample.

## Positioning vs siblings

- **learn-classification-regression** (ds d2): the labeled half of the zoo. This is the unlabeled half.
- **learn-anomaly-detection** (ds d2, Lumen): unsupervised anomaly methods live THERE - u9 points, never re-teaches.
- **learn-recommendation** (ds d2, Kirana): recommenders live there; u8's association rules stop at cross-sell tables.
- **learn-feature-engineering** (ds d3): feature construction (incl. why RFM); this course consumes RFM, points back.
- **learn-intro-ml** (ds d2): assumed door; supervised vs unsupervised first defined there.

## Source universe (fetched 2026-08-11, URLs verified by research agent)

1. **Andrew Ng, ML Specialization course 3** (Unsupervised Learning, Recommenders, RL) - week 1 =
   clustering: k-means only (intuition, objective, initialization, choosing k). Ng stops at k-means;
   his second unsupervised pillar is anomaly detection (-> sibling course).
   https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning
2. **scikit-learn user guide 2.3 Clustering + 2.5 Decomposition + mixture models** - the algorithm
   comparison table (scalability / use case / geometry axes), evaluation chapter (silhouette = only
   common metric needing no ground truth), PCA (centers but does NOT scale), GMM
   (generalizes k-means to covariance structure). https://scikit-learn.org/stable/modules/clustering.html
3. **DataCamp "Unsupervised Learning in Python"** - ch1 k-means, ch2 hierarchical + t-SNE,
   ch3 PCA, ch4 NMF. https://www.datacamp.com/courses/unsupervised-learning-in-python
4. **Google ML Clustering course** (~110 min) - k-means only, heavy on similarity design +
   evaluating results. https://developers.google.com/machine-learning/clustering
5. **StatQuest videos** - hierarchical, k-means, DBSCAN, PCA (step-by-step + practical tips),
   t-SNE, UMAP. https://statquest.org/video_index.html
6. **mlxtend association-rules docs** (support/confidence/lift, apriori).
   https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

## Verified fact base (each confirmed with URL by the research pass)

| Fact | Status | Source |
|---|---|---|
| k-means minimizes within-cluster sum of squares (inertia); k-means++ is sklearn default init | Confirmed | sklearn clustering guide |
| sklearn n_init='auto' = ONE k-means++ run (old default 10 was for random init) - CALLOUT, old tutorials teach the stale default | Confirmed correction | sklearn KMeans doc |
| Feature scaling decides the outcome: sklearn's own demo, unscaled 35% vs scaled 96% accuracy | Confirmed | sklearn plot_scaling_importance |
| Silhouette: -1..1, higher better, needs no ground truth; elbow method "subjective and unreliable", fake elbows on uniform noise (Schubert 2023 "Stop using the elbow criterion") | Confirmed | sklearn + Wikipedia elbow method |
| DBSCAN: density-based, arbitrary shapes, noise label -1, no k but eps/min_samples crucial; transductive (cannot label new points) | Confirmed | sklearn DBSCAN |
| GMM: soft assignment (predict_proba); "generalizes k-means to incorporate covariance structure" - SAFE PHRASING (limit-case version is Bishop PRML 9.3.2; say "generalization", never "identical") | Confirmed | sklearn mixture |
| PCA: orthogonal components maximizing variance, ordered by explained variance; sklearn auto-centers but never scales - scaling is on you | Confirmed | sklearn decomposition |
| t-SNE: cluster sizes meaningless, inter-cluster distances may mean nothing, low perplexity makes noise look clustered | Confirmed | distill.pub/2016/misread-tsne |
| UMAP: own FAQ blesses UMAP-then-clustering "with care" - use density-based (HDBSCAN) not k-means; density NOT preserved | Confirmed nuance | umap-learn FAQ |
| Association rules: confidence = sup(A∪C)/sup(A) directional; lift 1 = independent, >1 positive; apriori/fpgrowth | Confirmed | mlxtend docs |
| RFM = standard retail segmentation feature set (classic 5x5x5) | Confirmed | Wikipedia RFM |
| k-means ALWAYS partitions - sklearn's assumptions demo shows confident wrong clusters on anisotropic/wrong-k data | Confirmed | sklearn plot_kmeans_assumptions |

## Session map + coverage

| # | Session | Covers | Sources |
|---|---------|--------|---------|
| u1 | The pile with no labels | Supervised vs unsupervised; the brief; RFM features; "imposed, then validated" framing | Ng w1 ✓, Google intro ✓, RFM ✓ |
| u2 | What "similar" means | Distance; the cents disaster; z-scaling; scale lever on the lab | Google similarity ✓, sklearn scaling demo ✓ |
| u3 | k-means from the inside | Lloyd's loop, inertia, k-means++, local optima, n_init callout | Ng w1 ✓, sklearn k-means ✓ |
| u4 | Picking k | Silhouette; elbow named-then-demoted; the k-trap anti-lever | sklearn evaluation ✓, elbow criticism ✓ |
| u5 | Beyond k-means | Hierarchical/dendrogram, DBSCAN, GMM soft assignment; comparison table thinking | sklearn 2.3 table ✓, StatQuest ◐ |
| u6 | Compressing the customer | PCA (variance explained, loadings); t-SNE/UMAP honesty split | sklearn 2.5 ✓, Distill ✓, UMAP FAQ ✓ |
| u7 | Naming and shipping segments | Profiling, personas, stability, the segment sheet artifact | Google evaluating ◐, practitioner canon |
| u8 | What sells together | Support/confidence/lift, apriori; cross-sell tables | mlxtend ✓ |
| u9 | When clustering lies | Noise still clusters; sklearn assumptions demo; when NOT to cluster; sibling pointers | sklearn assumptions ✓, elbow noise ✓ |
| u10 | Capstone | Full pipeline: scale -> validate k -> name -> action -> monitor | all ✓ recap |

Legend: ✓ taught to the 80% bar · ◐ introduced, depth stays with source/sibling.

## Honestly NOT covered (by design)

- Anomaly detection methods (-> learn-anomaly-detection); recommenders/NMF (-> learn-recommendation)
- Writing sklearn code (concepts + real in-browser lab instead; code stays with DataCamp/sklearn docs)
- HDBSCAN/OPTICS/spectral/BIRCH (named on the comparison card only), topic modeling, embeddings
- Certificates/assessments stay with official providers

## Simulator canon (cluster-live.js - ALL verified live in-browser 2026-08-11 BEFORE fan-out)

Widget: `<div id="cluster-live" data-mode="MODE"></div>` + `<script src="../assets/cluster-live.js?v=1">`.
REAL k-means++ (best-of-5 restarts by inertia), DBSCAN, spherical-GMM EM, Jacobi PCA, O(n²)
silhouette, apriori-lite rule mining - all computed live. 300 customers, seeded (mulberry32(42)),
5 planted segments; monetary arrives in CENTS from the billing export (the raw-unit trap).
Planted truth enables a "truth match" score - honesty rail on every mode says real projects never get one.

| Mode (page) | Canon |
|---|---|
| scale (u2) | raw: sil 0.53, truth 70% (wallet bands; gifters+lapsed merged) -> scaled: sil 0.68, truth 100% |
| iterate (u3) | k-means++ settles ~6 rounds, inertia 92; unlucky random start locks at inertia 278, truth 65% |
| pick-k (u4) | silhouette by k peaks at k=5 (0.68); trap button k=12 -> sil 0.34, 3 segments "too small to fund". k=3 0.64, k=4 0.66 (close - the read: peak + actionability, not decimals) |
| algos (u5) | two-moons: k-means 83% (straight border), DBSCAN eps .32/min 5 = 100% traces both moons, GMM 83% soft |
| pca (u6) | PC1 50%, PC2 37%, PC3 14% - PC1+PC2 = 86% variance |
| basket (u8) | 500 baskets: top rule white sneakers -> crew socks, sup 14%, conf 72%, lift 3.0x; midi dress -> leather belt lift 2.9x |
| noise (u9) | uniform noise 5-means: sil 0.31 vs real pile 0.68 - "it answered anyway" |
| capstone (u10) | ladder: raw 0.53/70% -> scaled 0.68/100% -> k=5 confirmed -> 5 named segments (VIP perks / steady upsell / gifter occasion nudges / fresh welcome / lapsed win-back) |

Segment names + campaigns: VIP regulars (55, F~22, $115) perks · Steady mid-basket (90, F~9, $70)
upsell · One-time gifters (60, F~1.2, $98) occasion nudges · Fresh arrivals (45, F~3, $45) welcome
flow · Lapsed big spenders (50, R~225d, $93) win-back.
