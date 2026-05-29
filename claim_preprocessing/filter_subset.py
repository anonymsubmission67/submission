import pandas as pd

df = pd.read_csv("data/claims_metadata.csv")

# Filtere nur Republican und Democrat
df_filtered = df[df["party"].isin(["Republican", "Democrat"])].copy()

print("Original distribution:")
print(df_filtered.groupby(["party", "label"]).size().unstack(fill_value=0))

# Für jedes Label die kleinere Anzahl finden und beide Parteien auf diese Anzahl balancieren
balanced_claim_ids = set()

for label in df_filtered["label"].unique():
    if pd.isna(label):
        continue
        
    # Daten für dieses Label
    label_data = df_filtered[df_filtered["label"] == label]
    
    # Anzahl pro Partei
    republican_count = len(label_data[label_data["party"] == "Republican"])
    democrat_count = len(label_data[label_data["party"] == "Democrat"])
    
    # Kleinere Anzahl nehmen
    min_count = min(republican_count, democrat_count)
    
    print(f"\nLabel '{label}': Republican={republican_count}, Democrat={democrat_count}, Min={min_count}")
    
    if min_count > 0:
        # Sample die kleinere Anzahl von beiden Parteien
        republican_sample = label_data[label_data["party"] == "Republican"].sample(n=min_count, random_state=42)
        democrat_sample = label_data[label_data["party"] == "Democrat"].sample(n=min_count, random_state=42)
        
        # Sammle die claim_ids der ausgewählten Claims
        balanced_claim_ids.update(republican_sample["claim_id"].tolist())
        balanced_claim_ids.update(democrat_sample["claim_id"].tolist())

# Füge "included_in_balanced" Spalte hinzu
df["included_in_balanced"] = df["claim_id"].isin(balanced_claim_ids)

print(f"\nTotal claims: {len(df)}")
print(f"Claims included in balanced dataset: {df['included_in_balanced'].sum()}")

# Zeige Verteilung der balancierten Claims
print("\nBalanced distribution:")
balanced_df = df[df["included_in_balanced"] == True]
print(balanced_df.groupby(["party", "label"]).size().unstack(fill_value=0))

# Speichere das ursprüngliche DataFrame mit der neuen Spalte
# df.to_csv("data/claims_metadata.csv", index=False)
print(f"\nUpdated claims_metadata.csv with 'included_in_balanced' column")