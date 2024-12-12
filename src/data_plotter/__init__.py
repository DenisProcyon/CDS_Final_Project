import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from types import SimpleNamespace

class DataPlotter:
    def __init__(self) -> None:
        sns.set_theme(style="whitegrid") 
        self.default_configs = SimpleNamespace(
            figsize=(12, 8),
            color="skyblue",
            title_fontsize=16,
            label_fontsize=14,
            tick_labelsize=12
        )

    def plot_barchart(self, data: pd.Series, title: str, xlabel: str, ylabel: str, figsize: tuple = None):
        """
        Plots barchart for given parameters
        """
        plt.figure(figsize=self.default_configs.figsize if figsize is None else figsize)

        ax = data.plot(kind='bar', color=self.default_configs.color, edgecolor='black')

        plt.title(title, fontsize=self.default_configs.title_fontsize, weight='bold')
        plt.xlabel(xlabel, fontsize=self.default_configs.label_fontsize)
        plt.ylabel(ylabel, fontsize=self.default_configs.label_fontsize)

        plt.xticks(rotation=45, fontsize=self.default_configs.tick_labelsize, ha='right')
        plt.yticks(fontsize=self.default_configs.tick_labelsize)

        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout() 
        plt.show()

    def plot_numerical_distribution(self, data: pd.DataFrame, columns: list[str]):
        """
        Plots numerical distribution
        """
        fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(self.default_configs.figsize[0], len(columns) * 5))
        axes = axes.flatten() if len(columns) > 1 else [axes]

        for ax, column in zip(axes, columns):
            if column not in data:
                ax.set_title(f"Column '{column}' not found", fontsize=self.default_configs.title_fontsize, weight='bold')
                ax.axis('off')
                continue
            if data[column].dtype not in ['float64', 'int64']:
                ax.set_title(f"Column '{column}' is not numeric", fontsize=self.default_configs.title_fontsize, weight='bold')
                ax.axis('off')
                continue

            sns.kdeplot(
                data[column].dropna(),
                ax=ax,
                linewidth=2,
                color=self.default_configs.color
            )
            ax.set_title(f"Distribution of '{column}'", fontsize=self.default_configs.title_fontsize, weight='bold')
            ax.set_xlabel("Value", fontsize=self.default_configs.label_fontsize)
            ax.set_ylabel("Density", fontsize=self.default_configs.label_fontsize)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def plot_all_categorical(self, data: pd.DataFrame):
        """
        Plots districution of categorical columns
        """
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns

        num_cols = len(categorical_columns)
        if num_cols == 0:
            print("No categorical columns found.")
            return

        fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(self.default_configs.figsize[0], num_cols * 5))
        axes = axes.flatten() if num_cols > 1 else [axes]

        for ax, col in zip(axes, categorical_columns):
            value_counts = data[col].value_counts()
            value_counts.plot(kind='bar', ax=ax, color=self.default_configs.color, edgecolor='black')

            ax.set_title(f"{col} Distribution", fontsize=self.default_configs.title_fontsize, weight='bold')
            ax.set_xlabel(col, fontsize=self.default_configs.label_fontsize)
            ax.set_ylabel("Count", fontsize=self.default_configs.label_fontsize)
            ax.tick_params(axis='x', labelrotation=45, labelsize=self.default_configs.tick_labelsize)
            ax.tick_params(axis='y', labelsize=self.default_configs.tick_labelsize)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
