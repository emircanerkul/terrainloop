# TerrainLoop

**Bumped-Terrain 4D Looped Compaction for 27B → 9B Language Models**

TerrainLoop is a technical proposal for compacting deep language models across the layer-depth axis (4D compaction). Instead of reusing identical matrices in a standard looped transformer or flattening weights onto rigid ternary plateaus, TerrainLoop pairs a shared 4-bit bumped terrain base with four small 1-bit/ternary modifier chips per pass, per-loop normalization, and cross-loop KV sharing.

## Read

- [Full paper (Markdown)](terrainloop-final-paper.md)
- [Full paper (PDF)](terrainloop-final-paper.pdf)
- [Executive summary](SUMMARY.md)
- [Website source](terrainloop/)

## Core claim

> Unrolling two shared 9B 4-bit terrain blocks with four small terrain-coupled 1-bit/ternary modifier chips, per-loop RMSMatch normalization, and cross-loop KV sharing preserves fine-grained weight geometry while delivering 27B-equivalent depth at ~6.75 GB of weight storage.

## Structure

```text
terrainloop/
├── .gitignore
├── README.md
├── SUMMARY.md
├── terrainloop-final-paper.md
├── terrainloop-final-paper.pdf
└── terrainloop/
    ├── index.html
    ├── paper.html
    ├── styles.css
    ├── paper-content.js
    └── public/
        ├── terrainloop-paper.md
        ├── terrainloop-paper.pdf
        └── summary.md
```
