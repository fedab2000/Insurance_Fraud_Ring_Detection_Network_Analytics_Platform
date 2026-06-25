import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

NUM_CLAIMS = 1000

FRAUD_RING_PROFILES = [
    {"ring_id": "RING_1", "num_claims": 13, "fraud_rate": 1.00},
    {"ring_id": "RING_2", "num_claims": 50, "fraud_rate": 0.65},
    {"ring_id": "RING_3", "num_claims": 24, "fraud_rate": 0.79},
    {"ring_id": "RING_4", "num_claims": 41, "fraud_rate": 0.94},
    {"ring_id": "RING_5", "num_claims": 6, "fraud_rate": 0.50},
]

NUM_FRAUD_RING_CLAIMS = sum(r["num_claims"] for r in FRAUD_RING_PROFILES)
NORMAL_CLAIMS = NUM_CLAIMS - NUM_FRAUD_RING_CLAIMS

cities = [
    "Toronto", "Mississauga", "Brampton", "Hamilton",
    "Kitchener", "Waterloo", "Cambridge", "London"
]

claim_types = ["Collision", "Injury", "Theft", "Comprehensive", "Glass"]

vehicle_makes = [
    "Toyota", "Honda", "Ford", "BMW",
    "Tesla", "Hyundai", "Mazda"
]

entities = []
claims = []
relationships = []
fraud_rings = []

entity_counter = 1
claim_number = 1


def new_entity(entity_type):
    global entity_counter
    entity_id = f"{entity_type[:3].upper()}_{entity_counter:06d}"

    entities.append({
        "entity_id": entity_id,
        "entity_type": entity_type,
        "entity_name": f"{entity_type}_{entity_counter}",
        "risk_role": entity_type
    })

    entity_counter += 1
    return entity_id


def new_claim_id():
    global claim_number
    claim_id = f"CLM_{claim_number:06d}"
    claim_number += 1
    return claim_id


def random_claim_date(max_days_back=730):
    return (
        datetime.today()
        - timedelta(days=random.randint(0, max_days_back))
    ).date()


def add_relationship(source, target, relationship_type, claim_id):
    relationships.append([
        source,
        target,
        relationship_type,
        claim_id
    ])


# ==========================================================
# NORMAL CLAIMS
# ==========================================================

for _ in range(NORMAL_CLAIMS):
    claim_id = new_claim_id()

    claimant = new_entity("Claimant")
    vehicle = new_entity("Vehicle")
    address = new_entity("Address")
    phone = new_entity("Phone")
    repair = new_entity("RepairShop")
    broker = new_entity("Broker")
    adjuster = new_entity("Adjuster")

    injury_reported = np.random.choice([0, 1], p=[0.75, 0.25])
    witness_present = np.random.choice([0, 1], p=[0.65, 0.35])

    clinic = new_entity("MedicalClinic") if injury_reported else None

    lawyer = (
        new_entity("Lawyer")
        if injury_reported and random.random() < 0.20
        else None
    )

    tow = (
        new_entity("TowTruckCompany")
        if random.random() < 0.30
        else None
    )

    witness = (
        new_entity("Witness")
        if witness_present
        else None
    )

    vehicle_value = random.randint(8000, 60000)

    raw_amount = round(np.random.lognormal(mean=9, sigma=0.55), 2)

    claim_amount = round(
        min(raw_amount, vehicle_value * random.uniform(0.05, 0.75)),
        2
    )

    claim_to_vehicle_ratio = round(claim_amount / vehicle_value, 2)

    claims.append({
        "claim_id": claim_id,
        "policy_id": f"POL_{random.randint(100000, 999999)}",
        "claimant_id": claimant,
        "vehicle_id": vehicle,
        "claim_amount": claim_amount,
        "vehicle_value": vehicle_value,
        "claim_to_vehicle_ratio": claim_to_vehicle_ratio,
        "claim_type": random.choice(claim_types),
        "claim_date": random_claim_date(730),
        "accident_city": random.choice(cities),
        "province": "ON",
        "vehicle_make": random.choice(vehicle_makes),
        "vehicle_year": random.randint(2010, 2025),
        "police_report": int(np.random.choice([0, 1], p=[0.45, 0.55])),
        "witness_present": int(witness_present),
        "injury_reported": int(injury_reported),
        "fraud_label": 0,
        "fraud_ring_id": None
    })

    add_relationship(claimant, claim_id, "filed_claim", claim_id)
    add_relationship(claimant, address, "lives_at", claim_id)
    add_relationship(claimant, phone, "uses_phone", claim_id)
    add_relationship(claim_id, vehicle, "involves_vehicle", claim_id)
    add_relationship(claim_id, repair, "used_repair_shop", claim_id)
    add_relationship(claim_id, broker, "policy_sold_by", claim_id)
    add_relationship(claim_id, adjuster, "handled_by", claim_id)

    if clinic:
        add_relationship(claim_id, clinic, "visited_clinic", claim_id)

    if lawyer:
        add_relationship(claim_id, lawyer, "retained_lawyer", claim_id)

    if tow:
        add_relationship(claim_id, tow, "used_tow_company", claim_id)

    if witness:
        add_relationship(claim_id, witness, "witnessed_by", claim_id)


# ==========================================================
# REALISTIC FRAUD / SUSPICIOUS RINGS
# ==========================================================

for ring_profile in FRAUD_RING_PROFILES:
    ring_id = ring_profile["ring_id"]
    num_ring_claims = ring_profile["num_claims"]
    target_fraud_rate = ring_profile["fraud_rate"]
    fraud_claims_in_ring = round(num_ring_claims * target_fraud_rate)

    shared_repair = new_entity("RepairShop")
    shared_clinic = new_entity("MedicalClinic")
    shared_lawyer = new_entity("Lawyer")
    shared_tow = new_entity("TowTruckCompany")
    shared_broker = new_entity("Broker")
    shared_address = new_entity("Address")
    shared_phone = new_entity("Phone")
    shared_witness = new_entity("Witness")

    ring_claimants = [
        new_entity("Claimant")
        for _ in range(num_ring_claims)
    ]

    ring_vehicles = [
        new_entity("Vehicle")
        for _ in range(num_ring_claims)
    ]

    total_amount = 0
    fraud_amount = 0

    fraud_positions = set(
        random.sample(range(num_ring_claims), fraud_claims_in_ring)
    )

    for i in range(num_ring_claims):
        claim_id = new_claim_id()
        is_fraud = 1 if i in fraud_positions else 0

        claimant = ring_claimants[i]
        vehicle = ring_vehicles[i]
        adjuster = new_entity("Adjuster")

        if is_fraud == 1:
            vehicle_value = random.randint(8000, 45000)
            claim_amount = random.randint(15000, 60000)
            injury_reported = 1
            witness_present = 1
            police_report = 1
        else:
            vehicle_value = random.randint(12000, 65000)
            claim_amount = random.randint(3000, 25000)
            injury_reported = int(np.random.choice([0, 1], p=[0.60, 0.40]))
            witness_present = int(np.random.choice([0, 1], p=[0.50, 0.50]))
            police_report = int(np.random.choice([0, 1], p=[0.30, 0.70]))

        claim_to_vehicle_ratio = round(claim_amount / vehicle_value, 2)

        total_amount += claim_amount

        if is_fraud == 1:
            fraud_amount += claim_amount

        claims.append({
            "claim_id": claim_id,
            "policy_id": f"POL_{random.randint(100000, 999999)}",
            "claimant_id": claimant,
            "vehicle_id": vehicle,
            "claim_amount": claim_amount,
            "vehicle_value": vehicle_value,
            "claim_to_vehicle_ratio": claim_to_vehicle_ratio,
            "claim_type": random.choice(["Collision", "Injury"]),
            "claim_date": random_claim_date(365),
            "accident_city": random.choice(cities),
            "province": "ON",
            "vehicle_make": random.choice(vehicle_makes),
            "vehicle_year": random.randint(2015, 2025),
            "police_report": police_report,
            "witness_present": witness_present,
            "injury_reported": injury_reported,
            "fraud_label": is_fraud,
            "fraud_ring_id": ring_id
        })

        add_relationship(claimant, claim_id, "filed_claim", claim_id)
        add_relationship(claimant, shared_address, "lives_at", claim_id)
        add_relationship(claimant, shared_phone, "uses_phone", claim_id)
        add_relationship(claim_id, vehicle, "involves_vehicle", claim_id)
        add_relationship(claim_id, shared_repair, "used_repair_shop", claim_id)
        add_relationship(claim_id, shared_broker, "policy_sold_by", claim_id)
        add_relationship(claim_id, adjuster, "handled_by", claim_id)

        if injury_reported == 1:
            add_relationship(claim_id, shared_clinic, "visited_clinic", claim_id)
            add_relationship(claim_id, shared_lawyer, "retained_lawyer", claim_id)

        add_relationship(claim_id, shared_tow, "used_tow_company", claim_id)

        if witness_present == 1:
            add_relationship(claim_id, shared_witness, "witnessed_by", claim_id)

    fraud_rings.append({
        "fraud_ring_id": ring_id,
        "ring_type": "Suspicious Auto Insurance Network",
        "num_claims": num_ring_claims,
        "fraud_claims": fraud_claims_in_ring,
        "fraud_rate": round(fraud_claims_in_ring / num_ring_claims, 2),
        "total_claim_amount": total_amount,
        "fraud_claim_amount": fraud_amount,
        "main_connector": shared_repair,
        "description": (
            "Synthetic suspicious network with shared repair shop, broker, "
            "tow company, address, phone, and selected shared clinic/lawyer/witness links."
        )
    })


# ==========================================================
# EXPORT
# ==========================================================

claims_df = pd.DataFrame(claims)
entities_df = pd.DataFrame(entities)

relationships_df = pd.DataFrame(
    relationships,
    columns=[
        "source_id",
        "target_id",
        "relationship_type",
        "claim_id"
    ]
)

fraud_rings_df = pd.DataFrame(fraud_rings)

claims_df.to_csv(os.path.join(DATA_DIR, "claims.csv"), index=False)
entities_df.to_csv(os.path.join(DATA_DIR, "entities.csv"), index=False)

relationships_df.to_csv(
    os.path.join(DATA_DIR, "relationships.csv"),
    index=False
)

fraud_rings_df.to_csv(
    os.path.join(DATA_DIR, "fraud_rings.csv"),
    index=False
)

print("Synthetic fraud ring data created successfully.")
print(f"Claims: {len(claims_df)}")
print(f"Normal claims: {NORMAL_CLAIMS}")
print(f"Suspicious ring claims: {NUM_FRAUD_RING_CLAIMS}")
print(f"Entities: {len(entities_df)}")
print(f"Relationships: {len(relationships_df)}")
print(f"Fraud rings: {len(fraud_rings_df)}")