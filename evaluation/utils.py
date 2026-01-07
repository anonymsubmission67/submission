import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

PROMPT_TYPE = "all"


PERSONAS_PATH = "data/persona_balanced_metadata.csv"

target_llms = ["phi","liberal", "conservative",  "mistral", "mixtral", "qwen", "qwen_big", "llama3", "llama4","gpt-oss", "gpt4","deepseek_small", "deepseek", "grok_small",  "grok"]
target_llms_rebuttal = ["mistral", "llama3", "qwen", "deepseek_small","phi","liberal", "conservative", "qwen_big", "gpt-oss"]
opinion_llms = [] 

llm_colors = {
    'mistral': {"fillcolor": "white", "edgecolor": "gold"},
    'mixtral': {"fillcolor": "white", "edgecolor": "darkorange"},

    'llama3': {"fillcolor": "white", "edgecolor": "limegreen"},
    'llama4': {"fillcolor": "white", "edgecolor": "darkgreen"},
    
    'qwen': {"fillcolor": "white", "edgecolor": "chocolate"},
    'qwen_big': {"fillcolor": "white", "edgecolor": "saddlebrown"},

    'deepseek_small': {"fillcolor": "white", "edgecolor": "mediumpurple"},    
    'deepseek': {"fillcolor": "white", "edgecolor": "rebeccapurple"},


    'gpt-oss': {"fillcolor": "white", "edgecolor": "darkturquoise"},
    'gpt4': {"fillcolor": "white", "edgecolor": "teal"},
    
    "grok_small": {"fillcolor": "white", "edgecolor": "violet"},
    'grok': {"fillcolor": "white", "edgecolor": "magenta"},

    'phi': {"fillcolor": "white", "edgecolor": "gray"},

    'conservative': {"fillcolor": "white", "edgecolor": "red"},
    'liberal': {"fillcolor": "white", "edgecolor": "blue"},
    "american": {"fillcolor": "white", "edgecolor": "black"},

    'men': {"fillcolor": "orange", "edgecolor": "orange"},
    'women': {"fillcolor": "darkred", "edgecolor": "darkred"},
    'teenagers': {"fillcolor": "purple", "edgecolor": "purple"},
    'over_30': {"fillcolor": "pink", "edgecolor": "pink"},
    'old_people': {"fillcolor": "gray", "edgecolor": "gray"},

    'qwen_right': {"fillcolor": "white", "edgecolor": "gray"},
}

llm_info = {
    "mistral": {"source": "mistralai/Mistral-7B-Instruct-v0.1", "size_B": 7, "name": "Mistral 7B", "license": "Apache-2.0"},
"mixtral": {"source": "mistralai/Mixtral-8x7B-Instruct-v0.1", "size_B": 56, "name": "Mixtral 8x7B", "license": "Apache-2.0"},

"qwen": {"source": "Qwen/Qwen3-4B-Instruct-2507", "size_B": 4, "name": "Qwen3 4B", "license": "Apache-2.0"},
"qwen_big": {"source": "Qwen/Qwen3-30B-A3B-Instruct-2507", "size_B": 72, "name": "Qwen3 30B", "license": "Apache-2.0"},

"llama3": {"source": "meta-llama/Meta-Llama-3-8B-Instruct", "size_B": 8, "name": "Llama3 8B", "license": "llama3"},
"llama4": {"source": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "size_B": 109, "name": "Llama 4 Scout", "license": "llama4"}, # active: 17B

"gpt-oss": {"source": "openai/gpt-oss-20b", "size_B": 20, "name": "GPT-OSS 20B", "license": "Apache-2.0"},
"gpt4": {"source": "https://openrouter.ai/openai/gpt-4.1", "size_B": 200, "name": "GPT 4.1", "license": "paid service"}, # nur eine Schätzung für GPT-4o https://aiexpjourney.substack.com/p/the-number-of-parameters-of-gpt-4o

"deepseek_small": {"source": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "size_B": 32, "name": "DeepSeek R1 Dist.", "license": "mit"},
"deepseek": {"source": "https://openrouter.ai/deepseek/deepseek-chat-v3.1:free", "size_B": 671, "name": "DeepSeek 3.1", "license": "paid service"},

"grok_small": {"source": "https://openrouter.ai/x-ai/grok-3", "size_B": 500, "name": "Grok 3", "license": "paid service"}, # estimated Grok-1 had 314B
"grok": {"source": "https://openrouter.ai/x-ai/grok-4-fast:free", "size_B": 1700, "name": "Grok 4", "license": "paid service"}, # mehrere quellen sagen 1.7 trillion e.g. https://www.datastudios.org/post/grok-4-vs-previous-models-1-1-5-2-3-3-5-full-comparison-of-architecture-capabilities-and-r#:~:text=Grok%204%20(July%202025)%20represents,a%20huge%20jump%20in%20capacity.


"phi": {"source": "unsloth/Phi-3-mini-4k-instruct", "size_B": 3.8, "name": "Phi 3 mini 4K", "license": "access upon request"},
"conservative": {"source": "Opinion-GPT/opiniongpt-phi3-conservative", "size_B": 3.8, "name": "Phi 3 Conservative", "license": "access upon request"},
"liberal": {"source": "Opinion-GPT/opiniongpt-phi3-liberal", "size_B": 3.8, "name": "Phi 3 Liberal", "license": "access upon request"},

}


def make_tertile(s):
    s = pd.to_numeric(s, errors="coerce")
    cats, bins = pd.qcut(s, q=3, labels=None, retbins=True, duplicates="drop")
    n = len(bins) - 1  # actual number of bins after dropping duplicates
    labels = ["low", "middle", "high"][:n]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)




claims = pd.read_csv("data/claims_metadata.csv").set_index("claim_id")
personas = pd.read_csv("data/personas_metadata.csv").set_index("id")

# Clean income then bin
personas["income_num"] = (
    personas["income"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
)

personas["age_bin"] = make_tertile(personas["age"])
personas["income_bin"] = make_tertile(personas["income_num"])


# Map dataset labels to numeric 0..5
label_to_num = {
    "pants-fire": 0, "false": 1, "mostly-false": 2,
    "half-true": 3, "mostly-true": 4, "true": 5,
}

def calc_me_diff(df):
        df_long = df.T.merge(claims[["party", "label"]], left_index=True, right_index=True, how="left")
        df_long["label"] = df_long["label"].map(label_to_num)

        # Subtract dataset_label from all columns except "party" and "dataset_label"
        for col in df_long.columns:
            if col not in ["party", "label"]:
                df_long[col] = df_long[col] - df_long["label"]

        df_long = df_long.drop(columns=["label"])

        bias_by_party = (
            df_long.groupby(["party"])
            .mean()
        ).T

        return bias_by_party["Republican"] - bias_by_party["Democrat"]


def calc_me_diff_personas(llm):

    me_diff_results = {}

    for prompt in ["1", "2"]:
        df = pd.read_csv(f"data/claim_matrices/{PROMPT_TYPE}/personas_{prompt}/{llm}_mean.csv", index_col=0)

        me_diff_results[f"prompt_{prompt}"] = calc_me_diff(df)

    # Create DataFrame with both prompts and persona metadata
    df_me_diff = pd.DataFrame(me_diff_results).merge(
        personas[["sex", "ethnicity", "education", "political_view", "age_bin", "income_bin"]], 
        left_index=True, 
        right_index=True, 
        how="left"
    )
    return df_me_diff


