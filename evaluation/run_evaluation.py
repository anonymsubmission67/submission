
from rq1_no_persona.run_no_persona_significance import run_no_persona_significance
from rq1_no_persona.build_no_persona_pbt_plot import build_no_persona_pbt_plot
from rq1_no_persona.build_no_persona_small import build_no_persona_small
from rq1_no_persona.build_no_persona_big import build_no_persona_big

from rq2_personas.run_persona_significance import run_persona_significance
from rq2_personas.build_box_plots import build_box_plots

from rq3_correlations.run_data_preperation import run_data_preperation
from rq3_correlations.build_correlation_plot import build_correlation_plot
from rq3_correlations.build_no_persona_metric_correlation import build_no_persona_metric_correlation
from rq3_correlations.build_overview_table import build_overview_table
from rq3_correlations.build_pba_plot import build_pba_plot
from rq3_correlations.build_persona_correlation_table import build_persona_correlation_table
from rq3_correlations.build_scatter_plots import build_scatter_plots


if __name__ == "__main__":
    run_no_persona_significance()
    build_no_persona_pbt_plot()
    build_no_persona_small()
    build_no_persona_big()
    run_persona_significance()
    build_box_plots()
    run_data_preperation()
    build_correlation_plot()
    build_no_persona_metric_correlation()
    build_overview_table()
    build_pba_plot()
    build_persona_correlation_table()
    build_scatter_plots()