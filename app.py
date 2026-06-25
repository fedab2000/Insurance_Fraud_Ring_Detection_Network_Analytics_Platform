import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Insurance Fraud Ring Network Analytics",
    page_icon="🕸️",
    layout="wide"
)

DATA_DIR = "data"
OUTPUT_DIR = "outputs"


@st.cache_data
def load_data():
    return {
        "claims": pd.read_csv(os.path.join(DATA_DIR, "claims.csv")),
        "entities": pd.read_csv(os.path.join(DATA_DIR, "entities.csv")),
        "relationships": pd.read_csv(os.path.join(DATA_DIR, "relationships.csv")),
        "fraud_rings": pd.read_csv(os.path.join(DATA_DIR, "fraud_rings.csv")),
        "node_metrics": pd.read_csv(os.path.join(OUTPUT_DIR, "node_metrics.csv")),
        "cluster_summary": pd.read_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv")),
        "component_summary": pd.read_csv(os.path.join(OUTPUT_DIR, "component_summary.csv")),
        "siu_queue": pd.read_csv(os.path.join(OUTPUT_DIR, "siu_investigation_queue.csv")),
    }


data = load_data()

claims = data["claims"]
entities = data["entities"]
relationships = data["relationships"]
fraud_rings = data["fraud_rings"]
node_metrics = data["node_metrics"]
cluster_summary = data["cluster_summary"]
component_summary = data["component_summary"]
siu_queue = data["siu_queue"]

st.title("🕸️ Insurance Fraud Ring Detection & Network Analytics Platform")

st.markdown(
    """
    This standalone project uses **unsupervised graph analytics**, **connected components**,
    **network centrality**, and **K-Means clustering** to surface suspicious insurance
    networks for SIU review.
    """
)

st.sidebar.header("Filters")

risk_order = ["Very High", "High", "Medium", "Worth Looking Into", "Low"]
available_tiers = [x for x in risk_order if x in siu_queue["risk_tier"].unique()]

selected_tiers = st.sidebar.multiselect(
    "Risk Tier",
    available_tiers,
    default=available_tiers
)

entity_types = sorted(node_metrics["entity_type"].dropna().unique())

selected_entity_types = st.sidebar.multiselect(
    "Entity Type",
    entity_types,
    default=entity_types
)

filtered_siu = siu_queue[siu_queue["risk_tier"].isin(selected_tiers)]
filtered_nodes = node_metrics[node_metrics["entity_type"].isin(selected_entity_types)]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Executive Dashboard",
        "SIU Queue",
        "Suspicious Networks",
        "High-Risk Entities",
        "K-Means Clusters",
        "Fraud Ring Validation",
    ]
)

with tab1:
    st.subheader("Executive Dashboard")

    suspicious_networks = siu_queue[siu_queue["risk_tier"].isin(["Very High", "High"])]
    known_fraud_claims = int(claims["fraud_label"].sum())
    total_claim_amount = claims["claim_amount"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", f"{len(claims):,}")
    c2.metric("Total Claim Amount", f"${total_claim_amount:,.0f}")
    c3.metric("Known Fraud Claims", f"{known_fraud_claims:,}")
    c4.metric("Suspicious Networks", f"{len(suspicious_networks):,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Entities", f"{len(entities):,}")
    c6.metric("Relationships", f"{len(relationships):,}")
    c7.metric("Avg Fraud Rate in SIU Queue", f"{siu_queue['fraud_rate'].mean():.1%}")
    c8.metric("Highest Risk Score", f"{siu_queue['risk_score'].max():.2f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        tier_counts = (
            siu_queue["risk_tier"]
            .value_counts()
            .reindex(risk_order)
            .dropna()
            .reset_index()
        )
        tier_counts.columns = ["risk_tier", "count"]

        fig = px.bar(
            tier_counts,
            x="risk_tier",
            y="count",
            title="Risk Tier Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            siu_queue,
            x="risk_score",
            nbins=30,
            title="SIU Risk Score Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Suspicious Networks")
    st.dataframe(
        siu_queue.sort_values("risk_score", ascending=False).head(10),
        use_container_width=True,
    )

with tab2:
    st.subheader("SIU Prioritization Queue")

    st.markdown(
        """
        Components are ranked using fraud rate, claim volume, connected claim amount,
        fraud claim count, and average claim-to-vehicle-value ratio.
        """
    )

    st.dataframe(
        filtered_siu.sort_values("risk_score", ascending=False),
        use_container_width=True,
    )

    fig = px.scatter(
        filtered_siu,
        x="claim_count",
        y="risk_score",
        size="total_claim_amount",
        color="risk_tier",
        hover_name="component_id",
        title="Claim Count vs Risk Score",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Suspicious Network Explorer")

    component_options = (
        siu_queue.sort_values("risk_score", ascending=False)["component_id"]
        .head(50)
        .tolist()
    )

    selected_component = st.selectbox(
        "Select a component",
        component_options,
    )

    selected_summary = siu_queue[siu_queue["component_id"] == selected_component]

    if not selected_summary.empty:
        row = selected_summary.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Tier", row["risk_tier"])
        c2.metric("Risk Score", f"{row['risk_score']:.2f}")
        c3.metric("Claim Count", f"{int(row['claim_count']):,}")
        c4.metric("Fraud Rate", f"{row['fraud_rate']:.1%}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Total Claim Amount", f"${row['total_claim_amount']:,.0f}")
        c6.metric("Fraud Claims", f"{int(row['fraud_claim_count']):,}")
        c7.metric(
            "Avg Claim / Vehicle Ratio",
            f"{row['avg_claim_to_vehicle_ratio']:.2f}",
        )

    component_nodes = node_metrics[
        node_metrics["component_id"] == selected_component
    ].sort_values("degree_centrality", ascending=False)

    st.markdown("### Nodes in Selected Network")
    st.dataframe(component_nodes, use_container_width=True)

    st.markdown("### Entity Type Mix")
    entity_mix = component_nodes["entity_type"].value_counts().reset_index()
    entity_mix.columns = ["entity_type", "count"]

    fig = px.bar(
        entity_mix,
        x="entity_type",
        y="count",
        title="Entity Types in Selected Component",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("High-Risk Entity Analysis")

    sort_metric = st.selectbox(
        "Sort entities by",
        [
            "connected_claim_count",
            "total_connected_claim_amount",
            "fraud_claim_count",
            "avg_claim_to_vehicle_ratio",
            "degree_centrality",
            "betweenness_centrality",
            "eigenvector_centrality",
        ],
    )

    top_entities = (
        filtered_nodes
        .sort_values(sort_metric, ascending=False)
        .head(100)
    )

    st.dataframe(top_entities, use_container_width=True)

    entity_summary = (
        node_metrics
        .groupby("entity_type")
        .agg(
            node_count=("node_id", "count"),
            avg_connected_claims=("connected_claim_count", "mean"),
            avg_fraud_claims=("fraud_claim_count", "mean"),
            avg_claim_amount=("total_connected_claim_amount", "mean"),
            avg_degree=("degree_centrality", "mean"),
        )
        .reset_index()
        .sort_values("avg_connected_claims", ascending=False)
    )

    fig = px.bar(
        entity_summary,
        x="entity_type",
        y="avg_connected_claims",
        title="Average Connected Claims by Entity Type",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("K-Means Cluster Analysis")

    st.markdown(
        """
        K-Means groups nodes using engineered network features such as centrality,
        connected claims, connected claim amount, fraud claim count, and claim severity.
        """
    )

    st.dataframe(cluster_summary, use_container_width=True)

    fig = px.scatter(
        cluster_summary,
        x="avg_connected_claims",
        y="avg_fraud_claims",
        size="node_count",
        hover_name="kmeans_cluster",
        title="K-Means Cluster Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)

    selected_cluster = st.selectbox(
        "Select K-Means Cluster",
        sorted(node_metrics["kmeans_cluster"].unique()),
    )

    cluster_nodes = node_metrics[
        node_metrics["kmeans_cluster"] == selected_cluster
    ].sort_values("degree_centrality", ascending=False)

    st.dataframe(cluster_nodes.head(100), use_container_width=True)

with tab6:
    st.subheader("Fraud Ring Validation")

    st.markdown(
        """
        Since this is synthetic data, the embedded suspicious ring profiles are known.
        This tab compares the generated fraud-ring profiles with the networks surfaced
        by graph analytics.
        """
    )

    st.markdown("### Embedded Suspicious Ring Profiles")
    st.dataframe(fraud_rings, use_container_width=True)

    ring_summary = (
        claims[claims["fraud_ring_id"].notna()]
        .groupby("fraud_ring_id")
        .agg(
            claim_count=("claim_id", "count"),
            fraud_claims=("fraud_label", "sum"),
            total_claim_amount=("claim_amount", "sum"),
            avg_claim_amount=("claim_amount", "mean"),
            avg_claim_to_vehicle_ratio=("claim_to_vehicle_ratio", "mean"),
        )
        .reset_index()
    )

    ring_summary["fraud_rate"] = (
        ring_summary["fraud_claims"] / ring_summary["claim_count"]
    )

    st.markdown("### Generated Ring Claim Summary")
    st.dataframe(ring_summary, use_container_width=True)

    fig = px.bar(
        ring_summary,
        x="fraud_ring_id",
        y="claim_count",
        title="Claims per Embedded Suspicious Ring",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        ring_summary,
        x="fraud_ring_id",
        y="fraud_rate",
        title="Fraud Rate by Embedded Suspicious Ring",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top SIU Components")
    st.dataframe(
        siu_queue.sort_values("risk_score", ascending=False).head(10),
        use_container_width=True,
    )