# %%
# %load_ext autoreload
# %autoreload 2
# %%
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# %%
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
from utils.config_argparse import try_parse_args
import seaborn as sns

# %%
@dataclass
class Config:
    seed: int = 42
    model_name: str = "meta-llama/Llama-2-7b-hf"
    out_dir: str = './out_icml_2025'
    width: float = 6 # Column width for ICML

def calculate_accuracy_se(correct, total):
    """Calculate standard error for accuracy using binomial proportion."""
    p = correct / total
    se = np.sqrt((p * (1 - p)) / total)
    return se * 1.96  # 95% confidence interval

def plot_bars(ax, pivot_df, metric, error_metric, group_by_target, colors, primary_langs, secondary_langs, bar_width):
    legend_elements = {}
    for i, primary_lang in enumerate(primary_langs):
        x_offset = 0
        for j, secondary_lang in enumerate(secondary_langs):
            if primary_lang != secondary_lang:
                if group_by_target:
                    try:
                        height = pivot_df[metric][primary_lang][secondary_lang]
                    except KeyError:
                        height = np.nan
                    if error_metric:
                        try:
                            yerr = pivot_df[error_metric][primary_lang][secondary_lang]
                        except KeyError:
                            yerr = None
                    else:
                        yerr = None
                else:
                    try:
                        height = pivot_df[metric][secondary_lang][primary_lang]
                    except KeyError:
                        height = np.nan
                    if error_metric:
                        try:
                            yerr = pivot_df[error_metric][secondary_lang][primary_lang]
                        except KeyError:
                            yerr = None
                    else:
                        yerr = None
                
                if not pd.isna(height):
                    x = i + x_offset * bar_width
                    if yerr is not None and not pd.isna(yerr):
                        bar = ax.bar(x, height, bar_width, yerr=yerr, capsize=5, color=colors[j], alpha=0.8)
                    else:
                        bar = ax.bar(x, height, bar_width, color=colors[j], alpha=0.8)
                    if secondary_lang not in legend_elements:
                        legend_elements[secondary_lang] = bar[0]
                    x_offset += 1
    return legend_elements

def plot_language_performance(
    df, 
    filename=None, 
    group_by_target=False, 
    cfg=None,
    include_accuracy=True,
    include_acc_errors=True
):
    # Calculate accuracy standard error

    acc = df['correct'] / df['total']
    df['acc'] = acc
    df['acc_sem'] = np.sqrt((acc * (1 - acc)) / df['total'])
    
    # Create pivot table with necessary metrics
    pivot_df = df.pivot(
        index='src_lang', 
        columns='dest_lang', 
        values=['avg_prob', 'sem95_error', 'acc', 'acc_sem']
    )
    
    # Determine language groupings
    src_langs = pivot_df.index
    dest_langs = pivot_df.columns.get_level_values(1).unique()
    group_by_target = group_by_target  # Added for clarity

    # Set up figure based on whether to include accuracy plot
    if include_accuracy:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(cfg.width, cfg.width * 1.5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(cfg.width, cfg.width * 0.75))

    # Configure common plot parameters
    bar_width = 0.15
    colors = sns.color_palette(n_colors=len(dest_langs))
    primary_langs = dest_langs if group_by_target else src_langs
    secondary_langs = src_langs if group_by_target else dest_langs
    x_label = 'Target Language' if group_by_target else 'Source Language'

    # Plot probability metrics
    legend_elements = plot_bars(
        ax1, pivot_df, 'avg_prob', 'sem95_error', 
        group_by_target, colors, primary_langs, secondary_langs, bar_width
    )

    # Plot accuracy metrics if enabled
    if include_accuracy:
        error_metric = 'acc_sem' if include_acc_errors else None
        plot_bars(
            ax2, pivot_df, 'acc', error_metric,
            group_by_target, colors, primary_langs, secondary_langs, bar_width
        )

    # Configure axes and layout
    xtick_positions = np.arange(len(primary_langs)) + bar_width * (len(secondary_langs)/2 - 0.5)
    
    ax1.set_ylabel('Avg. Prob. of Correct Translation')
    ax1.set_ylim(0, 1)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.25)
    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels([])  # Hide x labels for top plot

    if include_accuracy:
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.25)
        ax2.set_xlabel(x_label)
        ax2.set_xticks(xtick_positions)
        ax2.set_xticklabels([lang.upper() for lang in primary_langs])
    else:
        ax1.set_xlabel(x_label)
        ax1.set_xticks(xtick_positions)
        ax1.set_xticklabels([lang.upper() for lang in primary_langs])

    # Add legend if we have elements
    if legend_elements:
        ax1.legend(
            legend_elements.values(), 
            [f'{lang.upper()}' for lang in legend_elements.keys()],
            loc='lower center', 
            ncol=len(legend_elements), 
            columnspacing=1.0
        )

    plt.tight_layout()

    # Save plot if config provided
    if cfg:
        out_dir = os.path.join(cfg.out_dir, 'plots', cfg.model_name.split("/")[-1])
        os.makedirs(out_dir, exist_ok=True)
        file_name = 'translation_combined.svg' if not filename else filename
        file_path = os.path.join(out_dir, file_name)
        plt.savefig(file_path, format='svg', bbox_inches='tight')
        print(f"Plot saved as {file_path}")

    plt.show()


def plot_language_performance_grid(df, filename = None, group_by_target=False, fig_size = (6,5), cfg=None):
    # Get unique latent languages
    latent_langs = df['latent_lang'].unique()
    
    # Set up the plot grid
    n_plots = len(latent_langs)
    n_cols = min(3, n_plots)  # Max 3 columns
    n_rows = (n_plots - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_size[0]*n_cols, fig_size[1]*n_rows), squeeze=False)
    
    # Flatten axes array for easy iteration
    axes_flat = axes.flatten()

    for idx, latent_lang in enumerate(latent_langs):
        ax = axes_flat[idx]
        
        # Filter data for current latent language
        df_latent = df[df['latent_lang'] == latent_lang]
        
        # Pivot the dataframe
        pivot_df = df_latent.pivot(index='src_lang', columns='dest_lang', values=['avg', 'sem95_error'])
        
        # Get source and destination languages
        src_langs = pivot_df.index.unique()
        dest_langs = pivot_df.columns.get_level_values(1).unique()

        # Determine primary and secondary languages based on grouping
        if group_by_target:
            primary_langs = dest_langs
            secondary_langs = src_langs
            x_label = 'Target Language'
            legend_prefix = 'from'
        else:
            primary_langs = src_langs
            secondary_langs = dest_langs
            x_label = 'Source Language'
            legend_prefix = 'to'

        # Set width of bars
        bar_width = 0.15
        
        # Generate a color palette
        colors = sns.color_palette(n_colors=len(secondary_langs))

        # Plot bars for each language pair
        for i, primary_lang in enumerate(primary_langs):
            x_offset = 0
            for j, secondary_lang in enumerate(secondary_langs):
                if primary_lang != secondary_lang:  # Skip same language pairs
                    if group_by_target:
                        height = pivot_df['avg'][primary_lang][secondary_lang]
                        yerr = pivot_df['sem95_error'][primary_lang][secondary_lang]
                    else:
                        height = pivot_df['avg'][secondary_lang][primary_lang]
                        yerr = pivot_df['sem95_error'][secondary_lang][primary_lang]
                    
                    if not pd.isna(height):
                        x = i + x_offset * bar_width
                        ax.bar(x, height, bar_width, yerr=yerr, capsize=5, 
                               color=colors[j], alpha=0.8)
                        x_offset += 1

        # Customize the subplot
        ax.set_ylabel('Probability of Correct Translation')
        ax.set_xlabel(x_label)
        ax.set_title(f'Latent Language: {latent_lang.upper()}')
        
        # Set xticks with offset
        xtick_positions = np.arange(len(primary_langs)) + bar_width * 1.5
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(primary_langs)

        # Set y-axis to start at 0 and end at 1
        ax.set_ylim(0, 1)

        # Add a grid for better readability
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    # Create a common legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(secondary_langs))]
    fig.legend(handles, [f'{legend_prefix} {lang.upper()}' for lang in secondary_langs], 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(secondary_langs))

    # Remove any unused subplots
    for j in range(idx+1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Adjust layout
    plt.tight_layout()
    
    # Save the plot as SVG
    if cfg:
        out_dir = os.path.join(cfg.out_dir, 'plots', cfg.model_name.split("/")[-1])
        os.makedirs(out_dir, exist_ok=True)
        file_name = 'translation_grid.svg' if not filename else filename
        file_path = os.path.join(out_dir, file_name)
        plt.savefig(file_path, format='svg')
        print(f"Plot saved as {file_path}")

    plt.show()

if __name__ == "__main__":
    cfg = Config()
    cfg = try_parse_args(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)
    vanilla_csv = os.path.join(cfg.out_dir, cfg.model_name.split("/")[-1], 
                              "translation_no_interv_latent.csv")
    df = pd.read_csv(vanilla_csv)
    
    # Example usage with different configurations
    plot_language_performance(df, cfg=cfg, filename="combined_metrics.svg",
                            include_accuracy=True, include_acc_errors=True)
    
    reject_csv = os.path.join(cfg.out_dir, cfg.model_name.split("/")[-1],
                                "translation_reject_latent.csv")
    
