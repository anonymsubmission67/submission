import pandas as pd
import numpy as np
import krippendorff

def evaluate_annotations():
    """
    Evaluate the topic annotations and calculate inter-rater reliability.
    """
    # Load the annotation data
    df = pd.read_csv('data/topic_annotation.csv')
    
    print("Topic Annotation Evaluation")
    print("=" * 50)
    
    # Display basic statistics
    print(f"Total topics annotated: {len(df)}")
    print(f"Annotators: {', '.join(df.columns[1:])}")
    

    # Work directly with the categorical data
    annotators = ['Jing', 'Daniel', 'Charlott']
    
    # Create reliability data directly from the dataframe
    reliability_data = []
    for annotator in annotators:
        rater_data = []
        for val in df[annotator]:
            if pd.isna(val):
                rater_data.append(None)
            else:
                rater_data.append(val)  # Keep as string
        reliability_data.append(rater_data)
    
    print(f"Reliability data format: {len(reliability_data)} raters, {len(reliability_data[0])} items")
    print(f"Sample data (first 5 items for each rater):")
    for i, annotator in enumerate(annotators):
        print(f"  {annotator}: {reliability_data[i][:5]}")
    
    # Check for missing values
    missing_data = []
    for i, annotator in enumerate(annotators):
        missing_count = sum(1 for val in reliability_data[i] if val is None)
        missing_data.append(missing_count)
        print(f"  {annotator}: {missing_count} missing")
    
    # Calculate Krippendorff's Alpha using the krippendorff package
    print(f"\nInter-rater Reliability Analysis:")
    print("-" * 30)
    
    # Data is already in the correct format for krippendorff
    print(f"Reliability data format: {len(reliability_data)} raters, {len(reliability_data[0])} items")
    print(f"Sample reliability data (first rater, first 5 items): {reliability_data[0][:5]}")
    
    # Calculate Krippendorff's Alpha
    try:
        # First, let's check the data more carefully
        print(f"\nData validation:")
        print(f"  Total items: {len(reliability_data[0])}")
        print(f"  Total raters: {len(reliability_data)}")
        
        # Check for complete cases (items with all raters)
        complete_cases = 0
        for i in range(len(reliability_data[0])):
            if all(reliability_data[j][i] is not None for j in range(len(reliability_data))):
                complete_cases += 1
        
        print(f"  Complete cases (all raters): {complete_cases}")
        
        # Calculate simple agreement rate first
        agreements = 0
        total_pairs = 0
        for i in range(len(reliability_data[0])):
            item_ratings = [reliability_data[j][i] for j in range(len(reliability_data)) if reliability_data[j][i] is not None]
            if len(item_ratings) >= 2:
                for j in range(len(item_ratings)):
                    for k in range(j+1, len(item_ratings)):
                        total_pairs += 1
                        if item_ratings[j] == item_ratings[k]:
                            agreements += 1
        
        if total_pairs > 0:
            simple_agreement = agreements / total_pairs
            print(f"  Simple pairwise agreement: {simple_agreement:.4f}")
        
        # Show category distribution
        all_ratings = []
        for rater_data in reliability_data:
            all_ratings.extend([r for r in rater_data if r is not None])
        
        from collections import Counter
        category_counts = Counter(all_ratings)
        print(f"  Category distribution: {dict(category_counts)}")
        
        alpha_nominal = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
        print(f"Krippendorff's Alpha (nominal): {alpha_nominal:.4f}")
        
        # Interpret the alpha value
        if alpha_nominal >= 0.8:
            interpretation = "Excellent agreement"
        elif alpha_nominal >= 0.67:
            interpretation = "Good agreement"
        elif alpha_nominal >= 0.4:
            interpretation = "Moderate agreement"
        else:
            interpretation = "Poor agreement"
        
        print(f"Interpretation: {interpretation}")
        
    except Exception as e:
        print(f"Error calculating Krippendorff's Alpha: {e}")
        alpha_nominal = np.nan
    
    # Calculate pairwise agreement
    print(f"\nPairwise Agreement:")
    print("-" * 20)
    
    annotators = df.columns[1:].tolist()
    for i, annotator1 in enumerate(annotators):
        for j, annotator2 in enumerate(annotators):
            if i < j:
                # Get valid pairs
                valid_mask = ~(pd.isna(df[annotator1]) | pd.isna(df[annotator2]))
                if valid_mask.sum() > 0:
                    agreement = (df[annotator1][valid_mask] == df[annotator2][valid_mask]).mean()
                    print(f"  {annotator1} vs {annotator2}: {agreement:.3f}")
    
    # Filter for unanimous agreement (all three annotators agree)
    print(f"\nUnanimous Agreement Analysis:")
    print("-" * 30)
    
    # Check for three-way agreement
    unanimous_mask = (
        (df[annotators[0]] == df[annotators[1]]) & 
        (df[annotators[1]] == df[annotators[2]]) &
        df[annotators[0]].notna() &
        df[annotators[1]].notna() &
        df[annotators[2]].notna()
    )
    
    unanimous_df = df[unanimous_mask].copy()
    print(f"Topics with unanimous agreement: {len(unanimous_df)} out of {len(df)} ({len(unanimous_df)/len(df)*100:.1f}%)")
    
    # Show distribution of unanimous categories
    if len(unanimous_df) > 0:
        print(f"\nDistribution of unanimous categories:")
        category_counts = unanimous_df[annotators[0]].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} topics ({count/len(unanimous_df)*100:.1f}%)")
        
        # Show some examples
        print(f"\nExamples of unanimous agreements:")
        print("-" * 35)
        for i, row in unanimous_df.head(10).iterrows():
            topic = row['topic']
            category = row[annotators[0]]
            print(f"  {topic}: {category}")
    
    # Show disagreements
    disagreements = df[~unanimous_mask & df[annotators[0]].notna() & df[annotators[1]].notna() & df[annotators[2]].notna()]
    if len(disagreements) > 0:
        print(f"\nTopics with disagreements ({len(disagreements)} topics):")
        print("-" * 45)
        for i, row in disagreements.head(10).iterrows():
            topic = row['topic']
            annotations = [row[col] for col in annotators]
            print(f"  {topic}: {annotations}")
    
    # Save unanimous agreements
    unanimous_df.to_csv('data/topic_annotation_unanimous.csv', index=False)
    print(f"\nUnanimous agreements saved to: data/topic_annotation_unanimous.csv")
    

    
    return alpha_nominal, unanimous_df

if __name__ == "__main__":
    alpha, unanimous_df = evaluate_annotations()
