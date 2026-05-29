import json
import pandas as pd
import hashlib
from data.party_mapping import PARTY_MAPPING


# Read all lines from the JSONL file and extract the required attributes
data = []
removed_count = 0

with open('data/politifact_scraped_new.jsonl', 'r') as f:
    for line_num, line in enumerate(f, 1):
        if line.strip():  # Skip empty lines
            try:
                item = json.loads(line.strip())
                
                # Check for None values in required fields
                claim_text = item.get('claim')
                speaker = item.get('speaker')
                verdict = item.get('verdict')
                url = item.get('url')
                topics = item.get('topics')
                
                # Skip if any required field is None
                if claim_text is None:
                    print(f"Line {line_num}: Skipping - claim is None")
                    continue
                if speaker is None:
                    print(f"Line {line_num}: Skipping - speaker is None")
                    continue
                if verdict is None:
                    print(f"Line {line_num}: Skipping - verdict is None")
                    continue
                if url is None:
                    print(f"Line {line_num}: Skipping - url is None")
                    continue
                if topics is None:
                    print(f"Line {line_num}: Skipping - topics is None")
                    continue
                
                # Check if claim starts with image/video references and skip these
                claim_lower = claim_text.lower().strip()
                if (claim_lower.startswith("video") or 
                    claim_lower.startswith("image") or 
                    claim_lower.startswith("videos") or 
                    claim_lower.startswith("images") or 
                    claim_lower.startswith("a video") or 
                    claim_lower.startswith("an image")):
                    print(f"Line {line_num}: Skipping - starts with image/video reference: {claim_text[:100]}...")
                    removed_count += 1
                    continue
                
                # Extract topics and join them into a single string
                topics_string = '; '.join(topics)
                
                # Create hash ID for the claim
                claim_hash = hashlib.md5(claim_text.encode()).hexdigest()[:8]
                
                # Extract year from URL (assuming format like /factchecks/2007/aug/01/...)
                year = ''
                if url:
                    try:
                        # Extract year from URL path
                        url_parts = url.split('/')
                        for part in url_parts:
                            if part.isdigit() and len(part) == 4:
                                year = int(part)
                                break
                    except:
                        year = ''
                
                # Extract the required attributes
                row = {
                    'claim_id': claim_hash,
                    'claim': claim_text,
                    'label': verdict,
                    'personality': speaker,
                    'url': url,
                    'year': year,
                    'tags': topics_string
                }
                
                data.append(row)
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error - {e}")
                continue
            except Exception as e:
                print(f"Line {line_num}: Unexpected error - {e}")
                continue

# Create DataFrame and save to CSV
df = pd.DataFrame(data)

df = df[df["year"] > 2022]

print(f"\nTotal claims removed due to image/video references: {removed_count}")
print(f"After filtering for year > 2022: {len(df)}")

# Filter and categorize by axis
social_tags = ["Women", "Sexuality", "Religion", "Race and Ethnicity", "LGTBQ+", "Immigration", "Guns", "Families", "Abortion", "Crime", "Children", "Civil Rights", "Drugs", "Disability", "Human Rights", "Homeless"]
economic_tags = ["Workers", "Trade", "Regulation","Retirement", "Taxes", "Environment", "Economy", "Energy", "Artificial intelligence", "Corporations", "Gas Prices", "Income", "Labor", "Technology", "Tourism"]

# Create axis column based on topics
def categorize_axis(topics_string):
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
df_filtered = df_reduced[df_reduced['axis'].isin(['social', 'economic'])]


print(f"\nFiltered to {len(df_filtered)} items with social or economic tags")
print(f"Social items: {len(df_filtered[df_filtered['axis'] == 'social'])}")
print(f"Economic items: {len(df_filtered[df_filtered['axis'] == 'economic'])}")

# # Show distribution of labels by axis
# print(f"\nLabel distribution by axis:")
# distribution = df_filtered.groupby(['axis', 'label']).size().unstack(fill_value=0)
# print(distribution)

# # Show percentages
# print(f"\nLabel distribution by axis (percentages):")
# distribution_pct = df_filtered.groupby(['axis', 'label']).size().groupby(level=0).apply(lambda x: x / x.sum() * 100).unstack(fill_value=0)
# print(distribution_pct.round(1))

# Print unique values of personality
unique_personalities = df_filtered['personality'].unique()

print(f"\nTotal unique personalities: {len(unique_personalities)}")

# Check which personalities cannot be assigned a party
print(f"\nPersonalities that cannot be assigned a party:")
unmapped_personalities = []
for personality in unique_personalities:
    if personality not in PARTY_MAPPING:
        unmapped_personalities.append(personality)

for i, personality in enumerate(unmapped_personalities, 1):
    print(f"{i:2d}. {personality}")

print(f"\nTotal unmapped personalities: {len(unmapped_personalities)}")

# Save unique personalities as DataFrame
df_personalities = pd.DataFrame({
    'personality': unmapped_personalities,
})
df_personalities.to_csv('data/politifact_personalities.csv', index=False)
print(f"\nPersonalities saved to: data/politifact_personalities.csv")


# Add party column to df_filtered
df_filtered['party'] = df_filtered['personality'].map(PARTY_MAPPING).fillna('Unknown')

# Filter for only Republican and Democrat personalities
df_party_filtered = df_filtered[df_filtered['party'].isin(['Republican', 'Democrat'])]

print(f"\nFiltered to {len(df_party_filtered)} items with Republican or Democrat personalities")
print(f"Republican items: {len(df_party_filtered[df_party_filtered['party'] == 'Republican'])}")
print(f"Democrat items: {len(df_party_filtered[df_party_filtered['party'] == 'Democrat'])}")

# Show distribution of labels by axis and party
print(f"\nLabel distribution by axis and party:")
distribution_party = df_party_filtered.groupby(['axis', 'party', 'label']).size().unstack(fill_value=0)
print(distribution_party)

# Show percentages by party
print(f"\nLabel distribution by party (percentages):")
distribution_party_pct = df_party_filtered.groupby(['party', 'label']).size().groupby(level=0).apply(lambda x: x / x.sum() * 100).unstack(fill_value=0)
print(distribution_party_pct.round(1))

# Save party-filtered data
df_party_filtered.to_csv('data/claims_metadata.csv', index=False)
