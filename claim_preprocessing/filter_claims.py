import json
import pandas as pd
import hashlib


df = pd.read_csv('data/politifact_processed_new.csv')

topic_annotations = pd.read_csv('data/topic_annotation.csv')

party_affiliations = pd.read_csv('data/party_affiliations.csv')

# Check for three-way agreement
unanimous_mask = (
    (topic_annotations["Charlott"] == topic_annotations["Jing"]) & 
    (topic_annotations["Jing"] == topic_annotations["Daniel"])
)
    
unanimous_topics = topic_annotations[unanimous_mask].copy()

print(unanimous_topics[unanimous_topics["Charlott"].isin(["s", "e"])])

social_tags = list(unanimous_topics[unanimous_topics["Charlott"] == "s"]["topic"])
economic_tags = list(unanimous_topics[unanimous_topics["Charlott"] == "e"]["topic"])

# Create axis column based on topics
def categorize_axis(topics_string):
    # Check if topics_string is NaN or None
    if pd.isna(topics_string) or topics_string is None:
        return "other"
    
    # Convert to string if it's not already
    topics_string = str(topics_string)
    
    has_social = any(topic in topics_string for topic in social_tags)
    has_economic = any(topic in topics_string for topic in economic_tags)
    
    if has_social and has_economic:
        return "other"
    elif has_social:
        return "social"
    elif has_economic:
        return "economic"
    else:
        return "other"


df['axis'] = df['tags'].apply(categorize_axis)


# Convert year to int, handling empty strings
df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

# Filter for recent years (adjust as needed based on your data)


# Filter to only social and economic items
df_filtered = df[df['axis'].isin(['social', 'economic'])]

df_filtered = df_filtered[(df_filtered['label'] != 'full-flop') & (df_filtered['label'] != 'half-flip')]

print(f"\nFiltered to {len(df_filtered)} items with social or economic tags")
print(f"Social items: {len(df_filtered[df_filtered['axis'] == 'social'])}")
print(f"Economic items: {len(df_filtered[df_filtered['axis'] == 'economic'])}")



# Create a mapping dictionary from the party_affiliations dataframe
party_mapping = dict(zip(party_affiliations['personality'], party_affiliations['Republican or Democrat or None']))

# Add party column to df_filtered
df_filtered['party'] = df_filtered['personality'].map(party_mapping).fillna('Unknown')


df_filtered = df_filtered[df_filtered['party'].isin(['Republican', 'Democrat'])]


df_filtered["label"] = df_filtered["label"].apply(lambda x: "mostly-false" if x == "barely-true" else x)

# Create boxplot showing distribution of claims per class for each party
import matplotlib.pyplot as plt
import numpy as np

# Count claims per label for each party
party_label_counts = df_filtered.groupby(['party', 'label']).size().reset_index(name='count')

# Prepare data for plotting
labels = ['false', 'mostly-false', 'half-true', 'mostly-true', 'true']
democrat_counts = []
republican_counts = []

for label in labels:
    dem_count = party_label_counts[(party_label_counts['party'] == 'Democrat') & 
                                  (party_label_counts['label'] == label)]['count'].iloc[0] if len(party_label_counts[(party_label_counts['party'] == 'Democrat') & (party_label_counts['label'] == label)]) > 0 else 0
    rep_count = party_label_counts[(party_label_counts['party'] == 'Republican') & 
                                  (party_label_counts['label'] == label)]['count'].iloc[0] if len(party_label_counts[(party_label_counts['party'] == 'Republican') & (party_label_counts['label'] == label)]) > 0 else 0
    
    democrat_counts.append(dem_count)
    republican_counts.append(rep_count)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, democrat_counts, width, label='Democrat', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, republican_counts, width, label='Republican', color='red', alpha=0.7)

ax.set_xlabel('Truthfulness Label', fontsize=12)
ax.set_ylabel('Number of Claims', fontsize=12)
# ax.set_title('Distribution of Claims by Party and Label (Before Balancing)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('output/images/politifact_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nDistribution before balancing:")
print(f"Democrat total: {sum(democrat_counts)}")
print(f"Republican total: {sum(republican_counts)}")

# Für jede Kombination aus party, axis und label die kleinere Anzahl finden
balanced_claim_ids = set()

for party in ["Republican", "Democrat"]:
    for axis in ["social", "economic"]:
        for label in df_filtered["label"].unique():
            if pd.isna(label):
                continue
                
            # Daten für diese Kombination
            combo_data = df_filtered[
                (df_filtered["party"] == party) & 
                (df_filtered["axis"] == axis) & 
                (df_filtered["label"] == label)
            ]
            
            if len(combo_data) == 0:
                continue
            
            # Finde die kleinere Anzahl zwischen den beiden Parteien für diese axis-label Kombination
            other_party = "Democrat" if party == "Republican" else "Republican"
            other_combo_data = df_filtered[
                (df_filtered["party"] == other_party) & 
                (df_filtered["axis"] == axis) & 
                (df_filtered["label"] == label)
            ]
            
            # Kleinere Anzahl nehmen
            min_count = min(len(combo_data), len(other_combo_data))
            
            print(f"\n{party}-{axis}-{label}: {len(combo_data)}, {other_party}-{axis}-{label}: {len(other_combo_data)}, Min={min_count}")
            
            if min_count > 0:
                # Sample die kleinere Anzahl von beiden Parteien
                sample1 = combo_data.sample(n=min_count, random_state=42)
                sample2 = other_combo_data.sample(n=min_count, random_state=42)
                
                # Sammle die claim_ids der ausgewählten Claims
                balanced_claim_ids.update(sample1["claim_id"].tolist())
                balanced_claim_ids.update(sample2["claim_id"].tolist())

# Füge "included_in_balanced" Spalte hinzu
df_filtered["included_in_balanced"] = df_filtered["claim_id"].isin(balanced_claim_ids)

print(f"\nTotal claims: {len(df_filtered)}")
print(f"Claims included in balanced dataset: {df_filtered['included_in_balanced'].sum()}")

# Zeige Verteilung der balancierten Claims
print("\nBalanced distribution:")
df_filtered = df_filtered[df_filtered["included_in_balanced"] == True]

df_filtered = df_filtered.drop("included_in_balanced", axis=1)
distribution_table = df_filtered.groupby(["party", "axis", "label"]).size().unstack(fill_value=0)

# Erstelle LaTeX-Tabelle mit Axis-Unterscheidung
latex_code = r"""
\begin{table}[h]
    \centering
    \caption{Balanced Distribution of Claims by Party, Axis, and Label}
    \label{tab:balanced_distribution}
    \begin{tabular}{@{}lcccc@{}}
        \toprule
        \multirow{2}{*}{\textbf{Label}} & \multicolumn{2}{c}{\textbf{Republican}} & \multicolumn{2}{c}{\textbf{Democrat}} \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5}
        & \textbf{Economic} & \textbf{Social} & \textbf{Economic} & \textbf{Social} \\
        \midrule
"""

# Füge Zeilen zur LaTeX-Tabelle hinzu
for label in ["false", "mostly-false", "half-true", "mostly-true", "true"]:
    if pd.isna(label):
        continue
    
    rep_economic = distribution_table.loc[('Republican', 'economic'), label] if ('Republican', 'economic') in distribution_table.index else 0
    rep_social = distribution_table.loc[('Republican', 'social'), label] if ('Republican', 'social') in distribution_table.index else 0
    dem_economic = distribution_table.loc[('Democrat', 'economic'), label] if ('Democrat', 'economic') in distribution_table.index else 0
    dem_social = distribution_table.loc[('Democrat', 'social'), label] if ('Democrat', 'social') in distribution_table.index else 0
    
    latex_code += f"        {label} & {rep_economic} & {rep_social} & {dem_economic} & {dem_social} \\\\ \n"

# Schließe die Tabellenstruktur
latex_code += r"""
        \bottomrule
    \end{tabular}
\end{table}
"""

# # Speichere die LaTeX-Tabelle
# latex_file_path = "output/tables/class_distribution.tex"
# with open(latex_file_path, "w") as f:
#     f.write(latex_code)

# print("LaTeX table saved to output/tables/balanced_distribution.tex")
# print("\nLaTeX code:")
# print(latex_code)

# # Speichere das ursprüngliche DataFrame mit der neuen Spalte
# df_filtered.to_csv("data/claims_metadata.csv", index=False)
# print(f"\nUpdated claims_metadata.csv with 'included_in_balanced' column")
