"""
engine/train_kge.py — Train real Knowledge Graph Embeddings using PyKEEN.

Trains TransE on 328K SPO triples from data/triples.parquet.
Saves entity + relation embeddings to data/kge_embeddings.pkl.

Run:
    python engine/train_kge.py                    # default: TransE, 128-dim, 100 epochs
    python engine/train_kge.py --model RotatE     # try RotatE
    python engine/train_kge.py --dim 64 --epochs 50  # faster training

On DGX Spark: trains on GPU (~3-5 min for 328K triples).
Locally: trains on CPU (~10-15 min).
"""
import argparse
import pickle
import time
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def load_triples():
    """Load SPO triples and prepare for PyKEEN."""
    triples_path = DATA / "triples.parquet"
    if not triples_path.exists():
        raise FileNotFoundError(f"Triples not found at {triples_path}. Run build_triples.py first.")

    df = pd.read_parquet(triples_path)
    print(f"Loaded {len(df):,} triples, {df['predicate'].nunique()} predicates")

    # PyKEEN expects columns: head, relation, tail (all strings)
    # Our schema: subject, predicate, object_val
    triples_df = df[["subject", "predicate", "object_val"]].copy()
    triples_df.columns = ["head", "relation", "tail"]

    # Drop any rows with NaN
    triples_df = triples_df.dropna()

    # PyKEEN can't handle very long strings — truncate object values
    triples_df["tail"] = triples_df["tail"].astype(str).str[:100]
    triples_df["head"] = triples_df["head"].astype(str).str[:100]

    print(f"  Unique heads: {triples_df['head'].nunique():,}")
    print(f"  Unique relations: {triples_df['relation'].nunique():,}")
    print(f"  Unique tails: {triples_df['tail'].nunique():,}")

    return triples_df


def train(model_name="TransE", embedding_dim=128, epochs=100):
    """Train KGE model and save embeddings."""
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    triples_df = load_triples()

    print(f"\nCreating TriplesFactory...")
    t0 = time.time()

    # Create TriplesFactory from DataFrame
    tf = TriplesFactory.from_labeled_triples(
        triples_df[["head", "relation", "tail"]].values,
    )
    print(f"  TriplesFactory: {tf.num_entities:,} entities, {tf.num_relations:,} relations, "
          f"{tf.num_triples:,} triples ({time.time() - t0:.1f}s)")

    # Split: 90% train, 5% validation, 5% test
    train_tf, valid_tf, test_tf = tf.split([0.9, 0.05, 0.05], random_state=42)

    print(f"\nTraining {model_name} (dim={embedding_dim}, epochs={epochs})...")
    t1 = time.time()

    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model=model_name,
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs, batch_size=1024),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=0.001),
        random_seed=42,
    )

    train_time = time.time() - t1
    print(f"  Training completed in {train_time:.1f}s")

    # Extract entity embeddings
    import torch
    model = result.model
    entity_to_id = tf.entity_to_id
    embeddings_dict = {}

    # Get all entity embeddings as numpy — handle different PyKEEN versions
    entity_rep = model.entity_representations[0]
    with torch.no_grad():
        indices = torch.arange(tf.num_entities)
        try:
            all_embeddings = entity_rep(indices).cpu().numpy()
        except Exception:
            # Fallback: try accessing the underlying embedding directly
            if hasattr(entity_rep, '_embeddings'):
                all_embeddings = entity_rep._embeddings.weight.data.cpu().numpy()
            elif hasattr(entity_rep, 'base'):
                all_embeddings = entity_rep.base(indices).cpu().numpy()
            else:
                all_embeddings = entity_rep(indices).detach().cpu().numpy()

    for label, idx in entity_to_id.items():
        embeddings_dict[label] = all_embeddings[idx]

    # Also extract relation embeddings
    relation_to_id = tf.relation_to_id
    relation_embeddings_dict = {}
    if hasattr(model, 'relation_representations') and len(model.relation_representations) > 0:
        rel_rep = model.relation_representations[0]
        with torch.no_grad():
            rel_indices = torch.arange(tf.num_relations)
            try:
                all_rel_emb = rel_rep(rel_indices).cpu().numpy()
            except Exception:
                if hasattr(rel_rep, '_embeddings'):
                    all_rel_emb = rel_rep._embeddings.weight.data.cpu().numpy()
                else:
                    all_rel_emb = rel_rep(rel_indices).detach().cpu().numpy()
        for label, idx in relation_to_id.items():
            relation_embeddings_dict[label] = all_rel_emb[idx]

    # Save
    out_path = DATA / "kge_embeddings.pkl"
    payload = {
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "num_entities": tf.num_entities,
        "num_relations": tf.num_relations,
        "num_triples": tf.num_triples,
        "entity_embeddings": embeddings_dict,
        "relation_embeddings": relation_embeddings_dict,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "train_time_s": train_time,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    mb = out_path.stat().st_size / 1e6
    print(f"\n=== KGE Training Complete ===")
    print(f"  Model     : {model_name}")
    print(f"  Dimensions: {embedding_dim}")
    print(f"  Entities  : {tf.num_entities:,}")
    print(f"  Relations : {tf.num_relations:,}")
    print(f"  Triples   : {tf.num_triples:,}")
    print(f"  Train time: {train_time:.1f}s")
    print(f"  Saved to  : {out_path} ({mb:.1f} MB)")

    # Quick sanity check
    resource_entities = [e for e in embeddings_dict if e.startswith("shelter_")]
    if resource_entities:
        import numpy as np
        e1 = embeddings_dict[resource_entities[0]]
        e2 = embeddings_dict[resource_entities[min(1, len(resource_entities)-1)]]
        cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        print(f"\n  Sanity: cosine_sim({resource_entities[0]}, {resource_entities[min(1, len(resource_entities)-1)]}) = {cos_sim:.3f}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KGE embeddings on NYC triples")
    parser.add_argument("--model", default="TransE", choices=["TransE", "RotatE", "ComplEx", "DistMult"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    train(model_name=args.model, embedding_dim=args.dim, epochs=args.epochs)