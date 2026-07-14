# Agora Plan Recommender

## State

Starter repo — single `main.py` with a PyCharm-generated `print_hi` stub. No build system, no tests, no CI, no dependencies, no package manifest.

## Execution

```bash
python main.py
```

## IDE

PyCharm project files live in `.idea/`. Keep them in sync if you add/modify project settings.

## Architecture

- **Algorithm**: LightGCN on a bipartite user–plan graph. Recommendations derive from co-consumption (same crowd attends plan A and B), not text similarity.
- **Semantic embeddings** (sentence-transformer): initialize graph embeddings before training and regularize during training. At inference, blend graph score and semantic score — semantic fills in when interaction data is sparse.
- **Cold-start plans**: embed description, find nearest neighbors in semantic space, average their graph embeddings.
- **Training is always offline** (notebooks). Backend only does inference — load weights, dot products.
- **Claude is the scraper only** — never in the recommendation loop. It extracts structured plans from raw HTML/API responses.
- **Image pipeline** (background removal, dominant color extraction) is a core UX concern.
- **Gallery, not marketplace**: no inventory, no transactions, no purchase flow. Plans link out.

## Directory layout (to be created)

```
agora/
├── backend/          # API routes, db, recommender, scraper, image pipeline
├── frontend/         # TBD — single HTML file to start
├── notebooks/        # offline training (embed plans, train LightGCN)
├── static/           # processed images
└── data/             # seed scripts, mock interactions
```

## Conventions

- Technology choices are open. When they arise, flag tradeoffs rather than assuming.
- This is a demo/research repo: clarity and runnability over production patterns.
