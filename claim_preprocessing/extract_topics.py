import json
import pandas as pd
import hashlib

df = pd.read_csv('data/politifact_processed_new.csv')

# Collect frequency of each topic
print(f"\nCollecting topic frequencies...")
all_topics = []
for topics_string in df['tags']:
    if pd.notna(topics_string) and topics_string.strip():  # Skip NaN and empty strings
        topics_list = topics_string.split('; ')
        all_topics.extend(topics_list)

# Count topic frequencies
topic_counts = pd.Series(all_topics).value_counts().sort_values(ascending=False)

# Create topic frequency table
topic_freq_df = pd.DataFrame({
    'topic': topic_counts.index,
    'frequency': topic_counts.values,
    'percentage': (topic_counts.values / len(df) * 100).round(2)
})

# Save topic frequency table
topic_freq_df.to_csv('data/topic_frequencies.csv', index=False)
print(f"Topic frequencies saved to: data/topic_frequencies.csv")

# Print summary statistics
print(f"\nTotal unique topics: {len(topic_freq_df)}")
print(f"Total topic mentions: {topic_freq_df['frequency'].sum()}")
print(f"Average topics per claim: {topic_freq_df['frequency'].sum() / len(df):.2f}")

print(f"\nTop 20 most frequent topics:")
print(topic_freq_df.head(20).to_string(index=False))

# Show topics that appear only once
single_occurrence = topic_freq_df[topic_freq_df['frequency'] == 1]
print(f"\nTopics appearing only once: {len(single_occurrence)}")
if len(single_occurrence) > 0:
    print("First 10 single-occurrence topics:")
    print(single_occurrence.head(10)['topic'].tolist())

