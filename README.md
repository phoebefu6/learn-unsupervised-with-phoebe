<!-- phoebe header -->

[![Open the live course](https://img.shields.io/badge/%E2%96%B6%20open%20the%20live%20course-1f6feb?style=for-the-badge)](https://phoebefu6.github.io/learn-unsupervised-with-phoebe/)
[![Star this repo](https://img.shields.io/github/stars/phoebefu6/learn-unsupervised-with-phoebe?style=for-the-badge&label=star%20this%20repo&color=444444)](https://github.com/phoebefu6/learn-unsupervised-with-phoebe/stargazers)
[![Free courses](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fphoebefu6.github.io%2Flearn-with-phoebe%2Fstats.json&query=%24.courses_live&label=free%20courses&style=for-the-badge&color=111111)](https://phoebefu6.github.io/learn-with-phoebe/)

### ▶︎ [Open the live course →](https://phoebefu6.github.io/learn-unsupervised-with-phoebe/)

Free, runs in your browser. No install, no login.

> 📚 Part of **[Learn with Phoebe](https://phoebefu6.github.io/learn-with-phoebe/)** - free, hands-on courses on AI, data, and the craft around them. **[Browse every course ↗](https://phoebefu6.github.io/learn-with-phoebe/)**

<!-- /phoebe header -->

# learn unsupervised with phoebe

Finding structure with no answer key. Ten builder sessions on clustering, dimensionality
reduction, and association rules - the unlabeled half of the machine learning zoo.

**Live site:** https://phoebefu6.github.io/learn-unsupervised-with-phoebe/

## The course

Mango Lane (fashion app) has 12,000 customers and no labels. Marketing wants segments it can
campaign to. Ten sessions get there honestly:

1. **The pile with no labels** - the mindset: clusters are imposed, then validated
2. **What "similar" means** - distance, the cents disaster, and the scaling lever (70% -> 100%)
3. **k-means from the inside** - Lloyd's loop stepped by hand, local optima live
4. **Picking k** - silhouette fluency, the elbow demoted, and the k=12 trap button
5. **Beyond k-means** - DBSCAN vs the moons, dendrograms, GMM's soft answers
6. **Compressing the customer** - real PCA, and why t-SNE/UMAP plots lie about distance
7. **Naming and shipping segments** - personas, stability, the segment sheet
8. **What sells together** - association rules mined live; lift is the honest column
9. **When clustering lies** - 5-means on pure noise answers confidently; the tells
10. **Capstone** - the four-press ladder from wallet bands to five named segments

The lab (`assets/cluster-live.js`) runs REAL algorithms in the browser: k-means++ (best-of-5
restarts), DBSCAN, spherical-GMM EM, Jacobi PCA, silhouette, apriori-lite rule mining - on a
deterministic simulated sample whose planted truth is the teaching instrument (and the pages
say so). Sources and coverage documented in `materials/official-course-map.md`.

by Phoebe Fu
