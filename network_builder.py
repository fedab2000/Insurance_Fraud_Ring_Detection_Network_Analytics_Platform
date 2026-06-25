import os
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLAIMS_FILE = os.path.join(DATA_DIR, "claims.csv")
ENTITIES_FILE = os.path.join(DATA_DIR, "entities.csv")
RELATIONSHIPS_FILE = os.path.join(DATA_DIR, "relationships.csv")


def load_data():
    claims = pd.read_csv(CLAIMS_FILE)
    entities = pd.read_csv(ENTITIES_FILE)
    relationships = pd.read_csv(RELATIONSHIPS_FILE)
    return claims, entities, relationships


def build_graph(entities, relationships):
    G = nx.Graph()

    for _, row in entities.iterrows():
        G.add_node(
            row["entity_id"],
            entity_type=row["entity_type"],
            entity_name=row.get("entity_name", ""),
            risk_role=row.get("risk_role", "")
        )

    for _, row in relationships.iterrows():
        # Keep adjusters in source data, but exclude them from component detection.
        if row["relationship_type"] == "handled_by":
            continue

        source = row["source_id"]
        target = row["target_id"]

        if not G.has_node(source):
            G.add_node(source, entity_type="Unknown")

        if not G.has_node(target):
            G.add_node(target, entity_type="Claim")

        G.add_edge(
            source,
            target,
            relationship_type=row["relationship_type"],
            claim_id=row["claim_id"]
        )

    return G


def add_component_ids(G):
    component_map = {}

    components = list(nx.connected_components(G))

    for i, component in enumerate(components):
        component_id = f"COMP_{i + 1:04d}"

        for node in component:
            component_map[node] = component_id

    return component_map


def calculate_node_metrics(G, claims, component_map):
    degree_centrality = nx.degree_centrality(G)

    betweenness_centrality = nx.betweenness_centrality(
        G,
        k=min(100, G.number_of_nodes()),
        seed=42
    )

    try:
        eigenvector_centrality = nx.eigenvector_centrality(
            G,
            max_iter=1000
        )
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = {
            node: 0 for node in G.nodes()
        }

    claim_amount_map = claims.set_index("claim_id")["claim_amount"].to_dict()
    fraud_label_map = claims.set_index("claim_id")["fraud_label"].to_dict()
    claim_ratio_map = claims.set_index("claim_id")[
        "claim_to_vehicle_ratio"
    ].to_dict()

    rows = []

    for node in G.nodes():
        neighbors = list(G.neighbors(node))

        connected_claims = [
            n for n in neighbors
            if str(n).startswith("CLM_")
        ]

        total_claim_amount = sum(
            claim_amount_map.get(claim_id, 0)
            for claim_id in connected_claims
        )

        fraud_claim_count = sum(
            fraud_label_map.get(claim_id, 0)
            for claim_id in connected_claims
        )

        avg_claim_to_vehicle_ratio = (
            sum(
                claim_ratio_map.get(claim_id, 0)
                for claim_id in connected_claims
            ) / len(connected_claims)
            if len(connected_claims) > 0
            else 0
        )

        rows.append({
            "node_id": node,
            "entity_type": G.nodes[node].get("entity_type", "Unknown"),
            "entity_name": G.nodes[node].get("entity_name", ""),
            "component_id": component_map.get(node),
            "num_neighbors": len(neighbors),
            "connected_claim_count": len(connected_claims),
            "total_connected_claim_amount": total_claim_amount,
            "fraud_claim_count": fraud_claim_count,
            "avg_claim_to_vehicle_ratio": avg_claim_to_vehicle_ratio,
            "degree_centrality": degree_centrality.get(node, 0),
            "betweenness_centrality": betweenness_centrality.get(node, 0),
            "eigenvector_centrality": eigenvector_centrality.get(node, 0)
        })

    return pd.DataFrame(rows)


def run_kmeans(node_metrics, n_clusters=5):
    features = [
        "num_neighbors",
        "connected_claim_count",
        "total_connected_claim_amount",
        "fraud_claim_count",
        "avg_claim_to_vehicle_ratio",
        "degree_centrality",
        "betweenness_centrality",
        "eigenvector_centrality"
    ]

    X = node_metrics[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    node_metrics["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = (
        node_metrics
        .groupby("kmeans_cluster")
        .agg(
            node_count=("node_id", "count"),
            avg_neighbors=("num_neighbors", "mean"),
            avg_connected_claims=("connected_claim_count", "mean"),
            avg_claim_amount=("total_connected_claim_amount", "mean"),
            avg_fraud_claims=("fraud_claim_count", "mean"),
            avg_claim_to_vehicle_ratio=(
                "avg_claim_to_vehicle_ratio",
                "mean"
            ),
            avg_degree=("degree_centrality", "mean"),
            avg_betweenness=("betweenness_centrality", "mean"),
            avg_eigenvector=("eigenvector_centrality", "mean")
        )
        .reset_index()
        .sort_values("avg_fraud_claims", ascending=False)
    )

    return node_metrics, cluster_summary


def create_component_summary(G, node_metrics, claims):
    component_rows = []

    for component_id, group in node_metrics.groupby("component_id"):
        nodes = group["node_id"].tolist()

        claim_nodes = [
            node for node in nodes
            if str(node).startswith("CLM_")
        ]

        claim_data = claims[
            claims["claim_id"].isin(claim_nodes)
        ]

        claim_count = len(claim_nodes)
        total_claim_amount = claim_data["claim_amount"].sum()
        fraud_claim_count = claim_data["fraud_label"].sum()

        avg_claim_to_vehicle_ratio = (
            claim_data["claim_to_vehicle_ratio"].mean()
            if claim_count > 0
            else 0
        )

        component_rows.append({
            "component_id": component_id,
            "node_count": len(nodes),
            "claim_count": claim_count,
            "total_claim_amount": total_claim_amount,
            "fraud_claim_count": fraud_claim_count,
            "fraud_rate": (
                fraud_claim_count / claim_count
                if claim_count > 0
                else 0
            ),
            "avg_claim_to_vehicle_ratio": avg_claim_to_vehicle_ratio
        })

    component_summary = pd.DataFrame(component_rows)

    return component_summary.sort_values(
        ["fraud_claim_count", "total_claim_amount"],
        ascending=False
    )


def create_siu_queue(component_summary):
    siu_queue = component_summary.copy()

    siu_queue = siu_queue[siu_queue["claim_count"] > 0].copy()

    siu_queue["risk_score"] = (
        siu_queue["fraud_rate"] * 35
        + siu_queue["fraud_claim_count"] * 1.25
        + siu_queue["claim_count"] * 0.45
        + (siu_queue["total_claim_amount"] / 100000) * 1.8
        + siu_queue["avg_claim_to_vehicle_ratio"] * 8
    )

    siu_queue["risk_score"] = siu_queue["risk_score"].round(2)

    siu_queue["risk_tier"] = pd.cut(
        siu_queue["risk_score"],
        bins=[-1, 5, 20, 50, 100, 9999],
        labels=[
            "Low",
            "Worth Looking Into",
            "Medium",
            "High",
            "Very High"
        ]
    )

    siu_queue = siu_queue.sort_values(
        "risk_score",
        ascending=False
    )

    return siu_queue


def main():
    claims, entities, relationships = load_data()

    G = build_graph(entities, relationships)

    print("Graph created.")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    component_map = add_component_ids(G)

    node_metrics = calculate_node_metrics(
        G,
        claims,
        component_map
    )

    node_metrics, cluster_summary = run_kmeans(
        node_metrics,
        n_clusters=5
    )

    component_summary = create_component_summary(
        G,
        node_metrics,
        claims
    )

    siu_queue = create_siu_queue(component_summary)

    node_metrics.to_csv(
        os.path.join(OUTPUT_DIR, "node_metrics.csv"),
        index=False
    )

    cluster_summary.to_csv(
        os.path.join(OUTPUT_DIR, "cluster_summary.csv"),
        index=False
    )

    component_summary.to_csv(
        os.path.join(OUTPUT_DIR, "component_summary.csv"),
        index=False
    )

    siu_queue.to_csv(
        os.path.join(OUTPUT_DIR, "siu_investigation_queue.csv"),
        index=False
    )

    print("Analysis complete.")
    print("Files created:")
    print("outputs/node_metrics.csv")
    print("outputs/cluster_summary.csv")
    print("outputs/component_summary.csv")
    print("outputs/siu_investigation_queue.csv")


if __name__ == "__main__":
    main()