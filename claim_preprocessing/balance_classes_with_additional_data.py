"""
Erweitert claims_metadata.csv um Claims aus dem balancierten 2020_2022-Datensatz,
um selteneren Labels aufzufüllen. Es werden immer Paare (1 Democrat + 1 Republican) eingefügt.
"""
import pandas as pd

# Bestehendes Dataset und Pool zusätzlicher Claims (balanciert, aus filter_claims.py)
df_existing = pd.read_csv("data/claims_metadata.csv")
df_pool = pd.read_csv("data/claims_metadata_2019_2022.csv")

# Spalte "period": alte Claims = "23_25", neu aus Phase 1/2 = "18_22"
df_existing["period"] = "23_25"

# Label "pants-fire" vor beiden Phasen entfernen
# df_existing = df_existing[df_existing["label"] != "pants-fire"].copy()
df_pool = df_pool[df_pool["label"] != "pants-fire"].copy()
print("Label 'pants-fire' aus bestehendem Dataset und Pool entfernt.")

# Label-Counts im bestehenden Dataset
label_counts = df_existing["label"].value_counts()
print("\nLabel-Counts im bestehenden Dataset (claims_metadata.csv):")
print(label_counts)

target_per_label = int(label_counts.max())
print(f"\nZiel: {target_per_label} Claims pro Label (größte Klasse)")

# Nur Claims aus dem Pool nutzen, die noch nicht im bestehenden Dataset sind
existing_ids = set(df_existing["claim_id"].astype(str))
df_new = df_pool[~df_pool["claim_id"].astype(str).isin(existing_ids)].copy()
print(f"Verfügbare neue Claims (nicht in claims_metadata): {len(df_new)}")

# Spalten anpassen: gleiche Reihenfolge wie claims_metadata
cols = [c for c in df_existing.columns if c in df_new.columns]
df_new = df_new[cols]

added_rows = []
used_new_ids = set()

# Immer Paare (1 Democrat + 1 Republican) einfügen, nicht einzelne Claims
for label in label_counts.index:
    current = int(label_counts[label])
    deficit = target_per_label - current
    if deficit <= 0:
        print(f"  {label}: bereits {current} (Ziel {target_per_label}), nichts zu ergänzen")
        continue

    # Nur in Paaren ergänzen: Anzahl Paare = deficit // 2
    pairs_to_add = deficit // 2
    if pairs_to_add == 0:
        print(f"  {label}: Defizit {deficit} -> 0 Paare (ungerade Ergänzung nicht möglich)")
        continue

    dem_candidates = df_new[
        (df_new["label"] == label)
        & (df_new["party"] == "Democrat")
        & (~df_new["claim_id"].astype(str).isin(used_new_ids))
    ]
    rep_candidates = df_new[
        (df_new["label"] == label)
        & (df_new["party"] == "Republican")
        & (~df_new["claim_id"].astype(str).isin(used_new_ids))
    ]

    n_pairs = min(pairs_to_add, len(dem_candidates), len(rep_candidates))
    if n_pairs == 0:
        print(f"  {label}: Defizit {deficit}, aber keine vollen Paare (Dem: {len(dem_candidates)}, Rep: {len(rep_candidates)})")
        continue

    sample_dem = dem_candidates.sample(n=n_pairs, random_state=42)
    sample_rep = rep_candidates.sample(n=n_pairs, random_state=42)
    used_new_ids.update(sample_dem["claim_id"].astype(str).tolist())
    used_new_ids.update(sample_rep["claim_id"].astype(str).tolist())
    added_rows.append(sample_dem)
    added_rows.append(sample_rep)
    print(f"  {label}: {current} -> +{n_pairs} Paare (+{2 * n_pairs} Claims) = {current + 2 * n_pairs}")

if added_rows:
    df_added = pd.concat(added_rows, ignore_index=True)
    df_added["period"] = "19_22"
    df_extended = pd.concat([df_existing, df_added], ignore_index=True)
else:
    df_extended = df_existing.copy()

# Ergebnis-Counts nach Phase 1 (Paare)
print("\nLabel-Counts nach Erweiterung (Phase 1 – Paare):")
print(df_extended["label"].value_counts())

# --- Phase 2: Von den übrigen Claims ausgeglichen (Democrat + Republican) pro Label hinzufügen ---
remainder = df_new[~df_new["claim_id"].astype(str).isin(used_new_ids)]

# Pro Label: wie viele Dem/Rep-Paare können wir noch hinzufügen? Begrenzt durch das Label mit den wenigsten Paaren.
labels_in_remainder = remainder["label"].unique()
n_pairs_per_label = {}
for label in labels_in_remainder:
    dem_count = len(remainder[(remainder["label"] == label) & (remainder["party"] == "Democrat")])
    rep_count = len(remainder[(remainder["label"] == label) & (remainder["party"] == "Republican")])
    n_pairs_per_label[label] = min(dem_count, rep_count)

n_pairs_phase2 = min(n_pairs_per_label.values()) if n_pairs_per_label else 0

print(f"\nÜbrige Claims im Pool: {len(remainder)}")
if n_pairs_phase2 > 0:
    print(f"Phase 2: Pro Label {n_pairs_phase2} Paare (Democrat + Republican) hinzufügen …")

    added_rows_phase2 = []
    for label in labels_in_remainder:
        dem_cand = remainder[
            (remainder["label"] == label) & (remainder["party"] == "Democrat")
        ]
        rep_cand = remainder[
            (remainder["label"] == label) & (remainder["party"] == "Republican")
        ]
        n_pairs = min(n_pairs_phase2, len(dem_cand), len(rep_cand))
        if n_pairs > 0:
            sample_dem = dem_cand.sample(n=n_pairs, random_state=43)
            sample_rep = rep_cand.sample(n=n_pairs, random_state=43)
            added_rows_phase2.append(sample_dem)
            added_rows_phase2.append(sample_rep)
            print(f"  {label}: +{n_pairs} Paare (+{2 * n_pairs} Claims)")

    if added_rows_phase2:
        df_phase2 = pd.concat(added_rows_phase2, ignore_index=True)
        df_phase2["period"] = "18_22"
        df_extended = pd.concat([df_extended, df_phase2], ignore_index=True)
        print(f"Phase 2: insgesamt +{len(df_phase2)} Claims hinzugefügt (ausgeglichen Dem/Rep).")
else:
    print("Keine weiteren Claims zum Ausbalancieren (übriger Pool oder keine vollen Paare pro Label).")

# Ergebnis-Counts (final)
print("\nLabel-Counts nach Erweiterung (final):")
print(df_extended["label"].value_counts())

# Tabelle: Anzahl Claims pro Party und Label (Kontrolle Ausgleich)
table_party_label = df_extended.groupby(["party", "label"]).size().unstack(fill_value=0)
print("\n--- Anzahl Claims pro Party und Label ---")
print(table_party_label)
print()
table_party_label.to_csv("data/claims_metadata_extended_party_label_counts.csv")
print("Tabelle gespeichert: data/claims_metadata_extended_party_label_counts.csv")

out_path = "data/claims_metadata_extended.csv"
df_extended.to_csv(out_path, index=False)
print(f"\nErweitertes Dataset gespeichert: {out_path} (n={len(df_extended)})")
