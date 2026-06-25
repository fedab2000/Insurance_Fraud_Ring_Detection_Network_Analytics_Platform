# Insurance Fraud Ring Detection & Network Analytics Platform

## Overview

The Insurance Fraud Ring Detection & Network Analytics Platform is a standalone Streamlit application designed to identify suspicious insurance fraud networks using graph analytics, network science, and unsupervised machine learning.

Unlike traditional fraud detection models that focus on individual claims, this project focuses on discovering relationships between claimants, vehicles, addresses, phone numbers, repair shops, medical clinics, lawyers, brokers, tow truck companies, witnesses, and other entities to identify potential organized fraud rings.

The platform demonstrates how graph-based analytics can support Special Investigation Units (SIU) by prioritizing suspicious networks for further investigation.

---

## Key Features

### Graph Analytics

* Network construction using claims and entity relationships
* Connected component detection
* Degree centrality analysis
* Betweenness centrality analysis
* Eigenvector centrality analysis

### Unsupervised Learning

* K-Means clustering of network entities
* Discovery of suspicious behavioral patterns
* Identification of high-risk entities

### Fraud Ring Detection

* Detection of connected suspicious networks
* Risk-based prioritization
* Fraud ring validation against known synthetic fraud labels

### Executive Dashboard

* Total claims
* Total claim amount
* Suspicious networks identified
* Fraud rate analytics
* SIU prioritization queue

---

## Technology Stack

* Python
* Streamlit
* Pandas
* NumPy
* NetworkX
* Scikit-Learn
* Plotly

---

## Project Structure

```text
insurance_fraud_ring_network/
│
├── app.py
├── network_builder.py
├── generate_synthetic_fraud_ring_data.py
│
├── data/
│   ├── claims.csv
│   ├── entities.csv
│   ├── relationships.csv
│   └── fraud_rings.csv
│
├── outputs/
│   ├── node_metrics.csv
│   ├── cluster_summary.csv
│   ├── component_summary.csv
│   └── siu_investigation_queue.csv
│
├── requirements.txt
└── README.md
```

---

## Data Model

### Claims

Each claim contains information such as:

* Claim ID
* Policy ID
* Claim Amount
* Vehicle Value
* Claim-to-Vehicle Ratio
* Claim Type
* Accident Location
* Vehicle Information
* Fraud Label

### Entities

The network includes:

* Claimants
* Vehicles
* Addresses
* Phone Numbers
* Repair Shops
* Medical Clinics
* Lawyers
* Tow Truck Companies
* Insurance Brokers
* Adjusters
* Witnesses

### Relationships

Examples include:

* Claimant filed claim
* Claimant lives at address
* Claimant uses phone
* Claim uses repair shop
* Claim visits clinic
* Claim retains lawyer
* Claim uses tow company
* Policy sold by broker

---

## Synthetic Fraud Rings

The platform generates realistic synthetic fraud networks with varying:

* Ring sizes
* Fraud rates
* Claim amounts
* Vehicle values
* Network complexity

Example ring profiles:

| Ring   | Claims | Fraud Rate |
| ------ | ------ | ---------- |
| RING_1 | 13     | 100%       |
| RING_2 | 50     | 65%        |
| RING_3 | 24     | 79%        |
| RING_4 | 41     | 94%        |
| RING_5 | 6      | 50%        |

---

## Risk Scoring

Components are prioritized using a composite risk score that considers:

* Fraud rate
* Fraud claim count
* Claim volume
* Total claim amount
* Average claim-to-vehicle ratio

Risk tiers include:

* Low
* Worth Looking Into
* Medium
* High
* Very High

---

## Streamlit Application

The application contains six major modules:

### Executive Dashboard

Provides an overview of claims activity, suspicious networks, and risk metrics.

### SIU Prioritization Queue

Ranks suspicious networks based on risk score.

### Suspicious Network Explorer

Allows investigators to review connected entities and network characteristics.

### High-Risk Entity Analysis

Identifies influential entities using graph centrality measures.

### K-Means Cluster Analysis

Groups entities with similar behavior patterns using unsupervised learning.

### Fraud Ring Validation

Compares detected suspicious networks against known synthetic fraud-ring profiles.

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate synthetic data:

```bash
python generate_synthetic_fraud_ring_data.py
```

Run network analytics:

```bash
python network_builder.py
```

Launch Streamlit:

```bash
streamlit run app.py
```

---

## Future Enhancements

Potential future enhancements include:

* Louvain community detection
* Graph Neural Networks (GNN)
* Isolation Forest anomaly detection
* Interactive network visualization with PyVis
* Real-time fraud scoring
* Guidewire ClaimCenter integration
* Azure deployment
* Databricks implementation

---

## Author

Feda Bashbishi, MBA, M.Sc. Engineering, MDSAI

University of Waterloo

This project was developed as a portfolio demonstration of fraud analytics, graph analytics, network science, business intelligence, and machine learning applied to property and casualty insurance fraud detection.
