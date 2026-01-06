import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add evaluation directory to path to allow importing utils
script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
# Annahme: merged_diff existiert bereits mit Spalten 'llm', 'prompt_type', 'slope_difference_economic', 'slope_difference_social'
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from utils import calc_me_diff, calc_me_diff_personas, target_llms, llm_colors, opinion_llms, llm_info, PROMPT_TYPE


def build_no_persona_pbt_plot():
    out_dir = Path("output/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")

    label_to_num = {
        "pants-fire": 0, "false": 1, "mostly-false": 2,
        "half-true": 3, "mostly-true": 4, "true": 5,
    }

    me_diff_results = {}
    for prompt in ["1", "2"]:

        hpc_path = f"data/claim_matrices/{PROMPT_TYPE}/no_persona_{prompt}_mean.csv"
        api_path = f"data/claim_matrices/{PROMPT_TYPE}/no_persona_api_{prompt}_mean.csv"
    
        df_hpc = pd.read_csv(hpc_path, index_col=0)
        df_api = pd.read_csv(api_path, index_col=0)
        df_long = pd.concat([df_hpc, df_api], axis=0)

        df_long = df_long.T.merge(claims[["party", "label", "axis"]], left_index=True, right_index=True, how="left")
        df_long["label"] = df_long["label"].map(label_to_num)

        # Subtract dataset_label from all columns except "party" and "dataset_label"
        for col in df_long.columns:
            if col not in ["party", "label", "axis"]:
                df_long[col] = df_long[col] - df_long["label"]

        df_long =df_long.drop(columns=["label"])


        bias_by_party = (
            df_long.groupby(["party", "axis"])
            .mean()
        ).T

        me_diff_economic = bias_by_party[("Republican", "economic")] - bias_by_party[("Democrat", "economic")]
        me_diff_social = bias_by_party[("Republican", "social")] - bias_by_party[("Democrat", "social")]

        me_diff_results[f"prompt_{prompt}"] = pd.DataFrame({
            "me_diff_economic": me_diff_economic,
            "me_diff_social": me_diff_social
        })

    # Create DataFrame with both prompts
    me_diff = pd.concat(me_diff_results, axis=1, keys=['prompt_1', 'prompt_2'])
    me_diff.columns = [f"{col[1]}_{col[0]}" for col in me_diff.columns]  # e.g., me_diff_economic_prompt_1


    # Asymmetric limits for better visualization
    x_min, x_max = -1.0, 0.5
    y_min, y_max = -1.0, 0.5

    plt.style.use('default')
    # Figur erstellen
    plt.figure(figsize=(8, 8))

    # Definiere Marker für die verschiedenen Prompt-Typen
    prompt_markers = {
        '1': 'o',    # Kreis für 'Prompt 1'
        '2': 's'   # Quadrat für 'Prompt 2'
    }

    # Plotte Daten für beide Prompt-Typen
    for prompt_type in ['1', '2']:
        for llm in target_llms + opinion_llms:
            

            # Filtere die Zeile für diesen LLM und Prompt-Typ
            if llm not in me_diff.index:
                continue
                
            x_val = me_diff.loc[llm, f"me_diff_economic_prompt_{prompt_type}"]
            y_val = me_diff.loc[llm, f"me_diff_social_prompt_{prompt_type}"]
            
            if pd.isna(x_val) or pd.isna(y_val):
                continue
                
            color = llm_colors.get(llm, 'gray')
            # marker = prompt_markers.get(prompt_type, 'o')
            
            # Different sizes for opinion_llms vs target_llms
            size = 70 if llm in opinion_llms else 70
            
            # Label nur für 'Prompt 1' für eine übersichtlichere Legende
            llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
            label = f"{llm_name} (Prompt {prompt_type})" if True else None
            
            facecolor = color["fillcolor"] if prompt_type == "1" else color["edgecolor"]

            plt.scatter(x_val, y_val,
                        label=label,
                        facecolor=facecolor,
                        edgecolor=color["edgecolor"],
                        linewidth=2,
                        marker='o',
                        s=size)

    # Achsen konfigurieren (gleich wie vorher)
    plt.grid(False)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Entferne Tick-Markierungen
    plt.tick_params(axis='both', which='both', length=0)
    plt.xticks([])
    plt.yticks([])

    # Achsen in der Mitte
    ax = plt.gca()

    # ax.set_facecolor("#F4F0F0")


    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Entferne alle Standard-Achsenlinien
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Alternative Definition mit annotate:
    plt.annotate('', xy=(x_min*0.9, 0), xytext=(0, 0), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    plt.annotate('', xy=(x_max*0.9, 0), xytext=(0, 0), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    plt.annotate('', xy=(0, y_min*0.9), xytext=(0, 0), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    plt.annotate('', xy=(0, y_max*0.9), xytext=(0, 0), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Beschriftungen
    plt.text(x_max*0.85, y_min*0.05, 'right', ha='center', va='center', fontsize=12)
    plt.text(x_min*0.85, y_min*0.05, 'left', ha='center', va='center', fontsize=12)
    plt.text(x_max*0.25, y_max*0.85, 'authoritarian', ha='center', va='center', fontsize=12)
    plt.text(x_max*0.20, y_min*0.85, 'libertarian', ha='center', va='center', fontsize=12)

    plt.text(x_max*0.45, y_min*0.03, 'economic', ha='center', va='center', fontsize=12)
    plt.text(x_max*0.03, y_min*0.5, 'social', ha='center', va='center', fontsize=12, rotation=90)  

    # Titel und Legende
    # plt.title('LLMs - Political Compass', fontsize=16)

    # Erstelle zwei separate Legenden und kombiniere sie
    # Erste Legende: Models
    models_legend_entries = []
    for llm in target_llms + opinion_llms:
        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        models_legend_entries.append(mlines.Line2D([], [], color=llm_colors[llm]["edgecolor"], marker='o', 
                                                linestyle='None', markersize=8, 
                                                label=llm_name))

    # Zweite Legende: Prompt Types
    prompt_legend_entries = [
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None', 
                    markersize=8, label='party-aware'),
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None', 
                    markersize=8, markerfacecolor='white', markeredgecolor='gray',
                    label='party-agnostic')
    ]

    # Erstelle die erste Legende (Models)
    legend1 = plt.legend(handles=models_legend_entries, loc='upper left', 
                        title="Models", framealpha=1, facecolor="white",
                        columnspacing=1.0, handletextpad=0.5)
    plt.gca().add_artist(legend1)

    # Erstelle die zweite Legende (Prompt Types) darunter mit gleicher Breite
    legend2 = plt.legend(handles=prompt_legend_entries, loc='upper left', 
                        bbox_to_anchor=(0, 0.55), title="Prompt Types", 
                        framealpha=1, facecolor="white", 
                        columnspacing=1.0, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig('output/images/political_compass_plot.png', format='png', dpi=300)
    # plt.show()

if __name__ == "__main__":
    build_no_persona_pbt_plot()