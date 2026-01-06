import pandas as pd
import json
import sys
from pathlib import Path

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import llm_info, target_llms


def build_overview_table():

    # Load the overview data
    with open("data/interim_results/overview_data.json", "r") as f:
        data = json.load(f)

    # Convert to DataFrame for easier manipulation
    df_data = []
    for item in data:
        row = {
            "llm": item["llm"],
            "persona": item["persona"],
            "pol_bias": item["pol_bias"],
            "me_diff_1": item["prompt_1"].get("me_diff", None),
            "f1_1": item["prompt_1"].get("f1", None),
            "me_diff_2": item["prompt_2"].get("me_diff", None),
            "f1_2": item["prompt_2"].get("f1", None)
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Get unique LLMs and personas
    # Sort LLMs according to target_llms order, then add any remaining LLMs not in target_llms
    available_llms = list(df["llm"].unique())
    llms = [llm for llm in target_llms if llm in available_llms]
    # Add any LLMs not in target_llms at the end (alphabetically sorted)
    remaining_llms = [llm for llm in available_llms if llm not in target_llms]
    llms.extend(sorted(remaining_llms))
    
    personas = ["no_persona", "democrat", "republican", "no_specific_political_view"]

    # Create the LaTeX table
    latex_content = r"""
    \begin{table}[h]
    \centering
    \caption{Overview of LLM Performance by Persona Type}
    \label{tab:overview}
    \rotatebox{90}{
    {\small
    \begin{tabular}{@{}l|ccccc|ccccc|ccccc|ccccc@{}}
    \toprule
    \multirow{2}{*}{\textbf{LLM}} & \multicolumn{5}{c|}{\textbf{No Persona}} & \multicolumn{5}{c|}{\textbf{Democrat Persona}} & \multicolumn{5}{c|}{\textbf{Republican Persona}} & \multicolumn{5}{c}{\textbf{Persona with No Political View}} \\
    \cmidrule(lr){2-6} \cmidrule(lr){7-11} \cmidrule(lr){12-16} \cmidrule(lr){17-21}
    & \textbf{PB-A} & \textbf{BP-T$_1$} & \textbf{F1$_1$} & \textbf{PB-T$_2$} & \textbf{F1$_2$} 
    & \textbf{PB-A} & \textbf{PB-T$_1$} & \textbf{F1$_1$} & \textbf{PB-T$_2$} & \textbf{F1$_2$} 
    & \textbf{PB-A} & \textbf{PB-T$_1$} & \textbf{F1$_1$} & \textbf{PB-T$_2$} & \textbf{F1$_2$} 
    & \textbf{PB-A} & \textbf{PB-T$_1$} & \textbf{F1$_1$} & \textbf{PB-T$_2$} & \textbf{F1$_2$} \\
    \midrule
    """

    # Add data rows
    for llm in llms:
        row_parts = [f"\\textbf{{{llm_info.get(llm, {}).get('name', llm.capitalize())}}}"]
        
        for persona in personas:
            # Get data for this LLM-persona combination
            llm_data = df[(df["llm"] == llm) & (df["persona"] == persona)]
            
            if len(llm_data) > 0:
                data_row = llm_data.iloc[0]
                
                # Format values (3 decimal places)
                pb = f"{data_row['pol_bias']:.3f}" if pd.notna(data_row['pol_bias']) else "---"
                me1 = f"{data_row['me_diff_1']:.3f}" if pd.notna(data_row['me_diff_1']) else "---"
                f11 = f"{data_row['f1_1']:.3f}" if pd.notna(data_row['f1_1']) else "---"
                me2 = f"{data_row['me_diff_2']:.3f}" if pd.notna(data_row['me_diff_2']) else "---"
                f12 = f"{data_row['f1_2']:.3f}" if pd.notna(data_row['f1_2']) else "---"
                
                row_parts.extend([pb, me1, f11, me2, f12])
            else:
                # No data available
                row_parts.extend(["---", "---", "---", "---", "---"])
        
        latex_content += " & ".join(row_parts) + " \\\\\n"

    # Close the table
    latex_content += r"""
    \bottomrule
    \end{tabular}
    }}
    \end{table}
    """

    # Save to file
    with open("output/tables/overview_table.tex", "w") as f:
        f.write(latex_content)

    print("Overview table saved to: output/tables/overview_table.tex")
    print(f"Table includes {len(llms)} LLMs and {len(personas)} persona types")

if __name__ == "__main__":
    build_overview_table()