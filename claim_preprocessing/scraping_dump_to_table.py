import json
import pandas as pd
import hashlib


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

df.to_csv('data/politifact_processed_new.csv', index=False)

print(f"\nTotal claims removed due to image/video references: {removed_count}")
print(f"After filtering for year > 2022: {len(df)}")

