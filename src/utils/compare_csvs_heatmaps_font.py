import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Set global font sizes to 18
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

class MultiSNRAudioEnhancementComparator:
    """
    Compare multiple audio enhancement methods across multiple SNR levels.
    Creates both per-SNR and amalgamated visualizations.
    """
    
    def __init__(self, csv_files_dict_template, snr_levels, output_folder=None, experiment_name=None):
        """
        Initialize comparator with multiple CSV files across SNR levels.
        
        Parameters:
        -----------
        csv_files_dict_template : dict
            Dictionary with method names as keys and CSV file path templates as values.
            Use {snr} as placeholder for SNR level (e.g., "path/file_{snr}dB.csv")
        snr_levels : list
            List of SNR levels to analyze (e.g., [-5, 0, 5, 10, 15])
        output_folder : str, optional
            Base output folder
        experiment_name : str, optional
            Name for this comparison experiment
        """
        self.snr_levels = sorted(snr_levels)
        self.method_names = list(csv_files_dict_template.keys())
        self.csv_template = csv_files_dict_template
        
        if len(self.method_names) < 2:
            raise ValueError("Must provide at least 2 methods for comparison")
        if len(self.method_names) > 14:
            raise ValueError("Maximum 8 methods can be compared")
        
        # Setup output folder
        if output_folder is None:
            try:
                script_dir = Path(__file__).parent
            except NameError:
                script_dir = Path.cwd()
            self.output_folder = script_dir / "results" / "compare_csv_multi_snr"
        else:
            self.output_folder = Path(output_folder)
        
        if experiment_name:
            self.experiment_name = experiment_name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = self.output_folder / f"{experiment_name}_{timestamp}"
        else:
            self.experiment_name = "multi_snr_comparison"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = self.output_folder / f"comparison_{timestamp}"
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {self.output_folder}")
        
        # Load all CSV files for all SNR levels
        self.dataframes = {}  # {snr: {method: df}}
        self.df_merged = {}  # {snr: merged_df}
        
        for snr in self.snr_levels:
            print(f"\n=== Loading SNR {snr}dB ===")
            self.dataframes[snr] = {}
            
            for method_name, template_path in csv_files_dict_template.items():
                # Replace {snr} placeholder with actual SNR value
                csv_path = template_path.replace('{snr}', str(snr))
                
                try:
                    self.dataframes[snr][method_name] = pd.read_csv(csv_path)
                    print(f"  Loaded {method_name}: {len(self.dataframes[snr][method_name])} rows")
                except FileNotFoundError:
                    print(f"  WARNING: File not found for {method_name} at {snr}dB: {csv_path}")
                    self.dataframes[snr][method_name] = None
                except Exception as e:
                    print(f"  ERROR loading {method_name} at {snr}dB: {str(e)}")
                    self.dataframes[snr][method_name] = None
        
        # Metrics to analyze
        self.metric_ranges = {
            'PESQ': (-0.5, 4.5),
            'SI_SDR': (-10, 30),
            'STOI': (0, 1),
            'DNSMOS_p808_mos': (1, 5),
            'DNSMOS_mos_sig': (1, 5),
            'DNSMOS_mos_bak': (1, 5),
            'DNSMOS_mos_ovr': (1, 5)
        }
        self.metrics = list(self.metric_ranges.keys())
        
        # Extended color palette for up to 8 methods
        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
                      '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        
        # Merge dataframes for each SNR
        for snr in self.snr_levels:
            self.df_merged[snr] = self._merge_dataframes(snr)
            self._extract_noise_types(snr)
    
    def _merge_dataframes(self, snr):
        """Merge all dataframes for a specific SNR level."""
        # Start with first available dataframe
        df_merged = None
        first_method = None
        
        for method_name in self.method_names:
            if self.dataframes[snr][method_name] is not None:
                df_merged = self.dataframes[snr][method_name].copy()
                first_method = method_name
                df_merged = df_merged.rename(columns={
                    metric: f'{metric}_{method_name}' 
                    for metric in self.metrics if metric in df_merged.columns
                })
                break
        
        if df_merged is None:
            print(f"WARNING: No data available for SNR {snr}dB")
            return pd.DataFrame()
        
        # Merge remaining dataframes
        for method_name in self.method_names:
            if method_name == first_method or self.dataframes[snr][method_name] is None:
                continue
            
            df_temp = self.dataframes[snr][method_name].copy()
            df_temp = df_temp.rename(columns={
                metric: f'{metric}_{method_name}' 
                for metric in self.metrics if metric in df_temp.columns
            })
            
            df_merged = df_merged.merge(
                df_temp, 
                on='clean_file', 
                suffixes=('', f'_{method_name}'),
                how='inner'
            )
        
        print(f"  Merged dataset for {snr}dB: {len(df_merged)} common files")
        return df_merged
    
    def _extract_noise_types(self, snr):
        """Extract noise types for a specific SNR level - Updated for NOIZEUS dataset."""
        df = self.df_merged[snr]
        
        if len(df) == 0:
            return
        
        # Find enhanced_file column
        enhanced_col = None
        for col in ['enhanced_file_x', 'enhanced_file', 'noisy_file']:
            if col in df.columns:
                enhanced_col = col
                break
        
        if enhanced_col is None:
            df['noise_category'] = 'Unknown'
            self.df_merged[snr] = df
            return
        
        def categorize_noise(filename):
            """Categorize based on NOIZEUS noise dataset."""
            filename_lower = str(filename).lower()
            
            # Babble/Speech
            if any(x in filename_lower for x in ['babble', 'cafeteria']):
                return 'Babble'
            
            # Train noises
            elif any(x in filename_lower for x in ['train', 'inside_train']):
                return 'Train'
            
            # Street/Traffic
            elif 'street' in filename_lower:
                return 'Street'
            
            # Car noises
            elif 'car' in filename_lower:
                return 'Car'
            
            # Construction
            elif any(x in filename_lower for x in ['construction', 'crane', 'drilling', 
                                                    'jackhammer', 'trucks_unloading']):
                return 'Construction'
            
            # Stationary/White noise
            elif any(x in filename_lower for x in ['fan', 'cooler', 'ssn', 'white', 'pc_fan']):
                return 'Stationary'
            
            # Flight
            elif 'flight' in filename_lower:
                return 'Flight'
            
            # Other
            else:
                return 'Other'
        
        df['noise_category'] = df[enhanced_col].apply(categorize_noise)
        self.df_merged[snr] = df
    
    def create_amalgamated_comparison_table(self):
        """
        Create comparison table showing all SNR levels.
        Returns DataFrame with SNR levels as rows and methods as columns for each metric.
        """
        results = []
        
        for metric in self.metrics:
            for snr in self.snr_levels:
                row = {'Metric': metric, 'SNR_dB': snr}
                
                for method_name in self.method_names:
                    method_col = f'{metric}_{method_name}'
                    
                    if len(self.df_merged[snr]) > 0 and method_col in self.df_merged[snr].columns:
                        avg_value = self.df_merged[snr][method_col].mean()
                        row[method_name] = round(avg_value, 4)
                    else:
                        row[method_name] = None
                
                results.append(row)
        
        return pd.DataFrame(results)
    
    def generate_text_report(self):
        """Generate comprehensive text report similar to the example format."""
        report_lines = []
        report_lines.append("=" * 120)
        report_lines.append(f"EXPERIMENT: {self.experiment_name.upper()}")
        report_lines.append("MULTI-SNR AUDIO ENHANCEMENT COMPARISON REPORT")
        report_lines.append("=" * 120)
        report_lines.append(f"Methods compared: {', '.join(self.method_names)}")
        report_lines.append(f"SNR levels analyzed: {', '.join([f'{snr}dB' for snr in self.snr_levels])}")
        
        # Get total number of files per SNR
        for snr in self.snr_levels:
            if len(self.df_merged[snr]) > 0:
                report_lines.append(f"Number of files at {snr}dB: {len(self.df_merged[snr])}")
        
        report_lines.append(f"Output folder: {self.output_folder}")
        report_lines.append("=" * 120)
        report_lines.append("")
        
        # Average metrics for each SNR level
        report_lines.append("AVERAGE METRIC VALUES BY SNR LEVEL:")
        report_lines.append("-" * 120)
        
        for snr in self.snr_levels:
            report_lines.append(f"\n>>> SNR = {snr}dB <<<")
            if len(self.df_merged[snr]) == 0:
                report_lines.append("  No data available")
                continue
            
            for metric in self.metrics:
                line = f"  {metric:>20s}:  "
                for method_name in self.method_names:
                    method_col = f'{metric}_{method_name}'
                    if method_col in self.df_merged[snr].columns:
                        avg_val = self.df_merged[snr][method_col].mean()
                        line += f"{method_name}={avg_val:.4f}  "
                report_lines.append(line)
        
        report_lines.append("\n" + "=" * 120)
        
        # IMPROVEMENT ANALYSIS - comparing all methods to first method (baseline)
        if len(self.method_names) > 1:
            baseline_method = self.method_names[0]
            comparison_methods = self.method_names[1:]
            
            report_lines.append("\nIMPROVEMENT OVER BASELINE:")
            report_lines.append(f"Baseline method: {baseline_method}")
            report_lines.append("Format: Value | (Absolute Change) | % (Percentage Change)")
            report_lines.append("-" * 120)
            
            for snr in self.snr_levels:
                if len(self.df_merged[snr]) == 0:
                    continue
                    
                report_lines.append(f"\n>>> SNR = {snr}dB <<<")
                
                for metric in self.metrics:
                    baseline_col = f'{metric}_{baseline_method}'
                    
                    if baseline_col not in self.df_merged[snr].columns:
                        continue
                    
                    baseline_val = self.df_merged[snr][baseline_col].mean()
                    line = f"  {metric:>20s}: {baseline_method}={baseline_val:.4f}"
                    
                    for comp_method in comparison_methods:
                        comp_col = f'{metric}_{comp_method}'
                        
                        if comp_col in self.df_merged[snr].columns:
                            comp_val = self.df_merged[snr][comp_col].mean()
                            abs_change = comp_val - baseline_val
                            
                            # Avoid division by zero
                            if abs(baseline_val) > 1e-6:
                                pct_change = (abs_change / abs(baseline_val)) * 100
                            else:
                                pct_change = 0
                            
                            sign = "+" if abs_change >= 0 else ""
                            line += f"  |  {comp_method}={comp_val:.4f} ({sign}{abs_change:.4f}, {sign}{pct_change:.2f}%)"
                    
                    report_lines.append(line)
            
            report_lines.append("\n" + "=" * 120)
            
            # FILE-LEVEL IMPROVEMENT STATISTICS
            report_lines.append("\nFILE-LEVEL IMPROVEMENT STATISTICS:")
            report_lines.append("-" * 120)
            
            for snr in self.snr_levels:
                if len(self.df_merged[snr]) == 0:
                    continue
                
                report_lines.append(f"\n>>> SNR = {snr}dB <<<")
                
                for metric in self.metrics:
                    baseline_col = f'{metric}_{baseline_method}'
                    
                    if baseline_col not in self.df_merged[snr].columns:
                        continue
                    
                    report_lines.append(f"\n{metric} (Range: {self.metric_ranges[metric][0]} to {self.metric_ranges[metric][1]}):")
                    
                    for comp_method in comparison_methods:
                        comp_col = f'{metric}_{comp_method}'
                        
                        if comp_col not in self.df_merged[snr].columns:
                            continue
                        
                        # Calculate per-file improvements
                        df = self.df_merged[snr]
                        differences = df[comp_col] - df[baseline_col]
                        
                        improved = (differences > 0).sum()
                        degraded = (differences < 0).sum()
                        unchanged = (differences == 0).sum()
                        total = len(differences)
                        
                        report_lines.append(f"  {comp_method}:")
                        report_lines.append(f"    Improved:  {improved}/{total} files ({100*improved/total:.1f}%)")
                        report_lines.append(f"    Degraded:  {degraded}/{total} files ({100*degraded/total:.1f}%)")
                        report_lines.append(f"    Unchanged: {unchanged}/{total} files ({100*unchanged/total:.1f}%)")
        
        report_lines.append("\n" + "=" * 120)
        
        # Performance by noise category
        report_lines.append("\nPERFORMANCE BY NOISE CATEGORY:")
        report_lines.append("-" * 120)
        
        # Get all categories
        all_categories = set()
        for snr in self.snr_levels:
            if 'noise_category' in self.df_merged[snr].columns:
                all_categories.update(self.df_merged[snr]['noise_category'].unique())
        
        for category in sorted(all_categories):
            report_lines.append(f"\n>>> {category} <<<")
            for metric in ['PESQ', 'SI_SDR', 'STOI', 'DNSMOS_mos_ovr']:
                report_lines.append(f"  {metric}:")
                for snr in self.snr_levels:
                    if len(self.df_merged[snr]) == 0:
                        continue
                    if 'noise_category' not in self.df_merged[snr].columns:
                        continue
                    
                    cat_data = self.df_merged[snr][self.df_merged[snr]['noise_category'] == category]
                    if len(cat_data) == 0:
                        continue
                    
                    line = f"    {snr}dB: "
                    for method_name in self.method_names:
                        method_col = f'{metric}_{method_name}'
                        if method_col in cat_data.columns:
                            avg_val = cat_data[method_col].mean()
                            line += f"{method_name}={avg_val:.3f}  "
                    report_lines.append(line)
        
        report_lines.append("\n" + "=" * 120)
        
        return "\n".join(report_lines)
    
    def plot_amalgamated_snr_comparison(self, figsize=(20, 14)):
        """
        Create comprehensive plot showing performance across all SNR levels (LINE GRAPH).
        One subplot for each key metric.
        """
        key_metrics = ['PESQ', 'SI_SDR', 'STOI', 'DNSMOS_p808_mos']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(key_metrics):
            ax = axes[idx]
            
            for i, method_name in enumerate(self.method_names):
                snr_values = []
                metric_values = []
                
                for snr in self.snr_levels:
                    method_col = f'{metric}_{method_name}'
                    
                    if len(self.df_merged[snr]) > 0 and method_col in self.df_merged[snr].columns:
                        avg_value = self.df_merged[snr][method_col].mean()
                        snr_values.append(snr)
                        metric_values.append(avg_value)
                
                if snr_values:
                    # Use modulo to handle more than 8 methods gracefully
                    color_idx = i % len(self.colors)
                    ax.plot(snr_values, metric_values, 
                           marker='o', linewidth=3, markersize=10,
                           label=method_name, color=self.colors[color_idx], alpha=0.8)
                    
                    # Add value labels
                    for snr, val in zip(snr_values, metric_values):
                        ax.annotate(f'{val:.3f}', 
                                   xy=(snr, val), 
                                   xytext=(0, 8),
                                   textcoords='offset points',
                                   ha='center', va='bottom',
                                   fontsize=14, fontweight='bold')
            
            ax.set_xlabel('SNR (dB)', fontsize=18, fontweight='bold')
            ax.set_ylabel(f'{metric} Score', fontsize=18, fontweight='bold')
            ax.set_title(f'{metric} Performance Across SNR Levels', fontsize=20, fontweight='bold')
            ax.legend(loc='best', fontsize=16)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(self.snr_levels)
        
        plt.suptitle(f'{self.experiment_name}: Performance Across SNR Levels', 
                     fontsize=22, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        return fig
    
    def plot_amalgamated_snr_bar_comparison(self, figsize=(22, 14)):
        """
        Create BAR CHART version showing performance across all SNR levels.
        One subplot for each key metric - emphasizes discrete SNR levels.
        """
        key_metrics = ['PESQ', 'SI_SDR', 'STOI', 'DNSMOS_p808_mos']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(key_metrics):
            ax = axes[idx]
            
            num_methods = len(self.method_names)
            num_snrs = len(self.snr_levels)
            width = 0.15  # Skinny bars
            x_pos = np.arange(num_snrs)
            
            for i, method_name in enumerate(self.method_names):
                metric_values = []
                
                for snr in self.snr_levels:
                    method_col = f'{metric}_{method_name}'
                    
                    if len(self.df_merged[snr]) > 0 and method_col in self.df_merged[snr].columns:
                        avg_value = self.df_merged[snr][method_col].mean()
                        metric_values.append(avg_value)
                    else:
                        metric_values.append(0)
                
                if metric_values:
                    color_idx = i % len(self.colors)
                    offset = (i - num_methods/2 + 0.5) * width
                    bars = ax.bar(x_pos + offset, metric_values, width,
                                 label=method_name, 
                                 color=self.colors[color_idx], 
                                 alpha=0.85,
                                 edgecolor='black',
                                 linewidth=1.5)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, metric_values):
                        if val != 0:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.3f}',
                                   ha='center', va='bottom',
                                   fontsize=12, fontweight='bold')
            
            ax.set_xlabel('SNR (dB)', fontsize=18, fontweight='bold')
            ax.set_ylabel(f'{metric} Score', fontsize=18, fontweight='bold')
            ax.set_title(f'{metric} Performance Across SNR Levels', fontsize=20, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{snr}' for snr in self.snr_levels])
            ax.legend(loc='best', fontsize=16)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        
        plt.suptitle(f'{self.experiment_name}: Performance Across SNR Levels (Bar Chart)', 
                     fontsize=22, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        return fig
    
    def plot_snr_noise_category_heatmap(self, metric='PESQ', figsize=(16, 10)):
        """
        Create heatmap showing performance by SNR and noise category.
        Separate heatmap for each method.
        """
        categories = set()
        for snr in self.snr_levels:
            if 'noise_category' in self.df_merged[snr].columns:
                categories.update(self.df_merged[snr]['noise_category'].unique())
        
        categories = sorted(list(categories))
        
        if not categories:
            print("No noise category information available")
            return None
        
        num_methods = len(self.method_names)
        fig, axes = plt.subplots(1, num_methods, figsize=figsize, sharey=True)
        
        if num_methods == 1:
            axes = [axes]
        
        # Get appropriate vmin/vmax for the metric
        vmin, vmax = self.metric_ranges.get(metric, (0, 5))
        
        for idx, method_name in enumerate(self.method_names):
            ax = axes[idx]
            
            # Create data matrix: rows=categories, columns=SNR levels
            data_matrix = np.zeros((len(categories), len(self.snr_levels)))
            
            for i, category in enumerate(categories):
                for j, snr in enumerate(self.snr_levels):
                    method_col = f'{metric}_{method_name}'
                    
                    if len(self.df_merged[snr]) > 0 and method_col in self.df_merged[snr].columns:
                        category_data = self.df_merged[snr][
                            self.df_merged[snr]['noise_category'] == category
                        ]
                        if len(category_data) > 0:
                            data_matrix[i, j] = category_data[method_col].mean()
                        else:
                            data_matrix[i, j] = np.nan
                    else:
                        data_matrix[i, j] = np.nan
            
            # Plot heatmap with metric-appropriate range
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(self.snr_levels)))
            ax.set_xticklabels([f'{snr}dB' for snr in self.snr_levels], fontsize=16)
            ax.set_yticks(np.arange(len(categories)))
            ax.set_yticklabels(categories, fontsize=16)
            
            # Add text annotations
            for i in range(len(categories)):
                for j in range(len(self.snr_levels)):
                    if not np.isnan(data_matrix[i, j]):
                        text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                                      ha="center", va="center", color="black",
                                      fontsize=10)
            
            ax.set_title(f'{method_name}', fontsize=18, fontweight='bold')
            ax.set_xlabel('SNR Level', fontsize=16)
            
            if idx == 0:
                ax.set_ylabel('Noise Category', fontsize=16)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'{metric} Performance: SNR vs Noise Category', 
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig
    
    def plot_snr_bar_comparison(self, snr, figsize=(20, 12)):
        """Create bar chart for a specific SNR level."""
        if len(self.df_merged[snr]) == 0:
            print(f"No data available for SNR {snr}dB")
            return None
        
        group1 = ['PESQ', 'SI_SDR']
        group2 = ['STOI']
        group3 = ['DNSMOS_p808_mos', 'DNSMOS_mos_sig', 'DNSMOS_mos_bak', 'DNSMOS_mos_ovr']
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.6, 1.2], hspace=0.4)
        
        ax1 = fig.add_subplot(gs[0])
        self._plot_metric_group(ax1, group1, f'PESQ & SI-SDR Comparison (SNR={snr}dB)', snr)
        
        ax2 = fig.add_subplot(gs[1])
        self._plot_metric_group(ax2, group2, f'STOI Comparison (SNR={snr}dB)', snr)
        
        ax3 = fig.add_subplot(gs[2])
        self._plot_metric_group(ax3, group3, f'DNSMOS Metrics Comparison (SNR={snr}dB)', snr)
        
        plt.suptitle(f'{self.experiment_name}: Multi-Method Comparison at {snr}dB', 
                     fontsize=22, fontweight='bold', y=0.995)
        
        return fig
    
    def _plot_metric_group(self, ax, metrics, title, snr):
        """Helper function to plot a group of metrics for specific SNR."""
        df = self.df_merged[snr]
        
        if len(df) == 0:
            return
        
        num_methods = len(self.method_names)
        x_pos = np.arange(len(metrics))
        width = 0.8 / num_methods
        
        all_data = []
        for method_name in self.method_names:
            method_avgs = []
            for metric in metrics:
                method_col = f'{metric}_{method_name}'
                if method_col in df.columns:
                    method_avgs.append(df[method_col].mean())
                else:
                    method_avgs.append(0)
            all_data.append(method_avgs)
        
        for i, (method_name, method_data) in enumerate(zip(self.method_names, all_data)):
            offset = (i - num_methods/2 + 0.5) * width
            color_idx = i % len(self.colors)
            bars = ax.bar(x_pos + offset, method_data, width, 
                         label=method_name, 
                         color=self.colors[color_idx], 
                         alpha=0.8, 
                         edgecolor='black')
            
            for bar in bars:
                height = bar.get_height()
                if height != 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', 
                           fontsize=14 if num_methods > 3 else 16, 
                           fontweight='bold')
        
        ax.set_xlabel('Metrics', fontsize=18, fontweight='bold')
        ax.set_ylabel('Average Value', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=0, ha='center')
        ax.legend(loc='upper left', fontsize=16, ncol=2 if num_methods > 3 else 1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    def export_all_results(self):
        """Export all results including per-SNR and amalgamated visualizations."""
        print("\n" + "="*80)
        print("EXPORTING RESULTS...")
        print("="*80)
        
        # 1. Amalgamated comparison table
        amalgamated_table = self.create_amalgamated_comparison_table()
        amalg_path = self.output_folder / f"{self.experiment_name}_amalgamated_all_snr.csv"
        amalgamated_table.to_csv(amalg_path, index=False)
        print(f"✓ Amalgamated table: {amalg_path}")
        
        # 2. Text report
        text_report = self.generate_text_report()
        report_path = self.output_folder / f"{self.experiment_name}_REPORT.txt"
        with open(report_path, 'w') as f:
            f.write(text_report)
        print(f"✓ Text report: {report_path}")
        
        # 3. Amalgamated SNR comparison plot (LINE GRAPH)
        fig_amalg = self.plot_amalgamated_snr_comparison()
        amalg_plot_path = self.output_folder / f"{self.experiment_name}_amalgamated_snr_LINE.png"
        fig_amalg.savefig(amalg_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig_amalg)
        print(f"✓ Amalgamated SNR plot (line): {amalg_plot_path}")
        
        # 4. Amalgamated SNR comparison plot (BAR CHART)
        # fig_bar_amalg = self.plot_amalgamated_snr_bar_comparison()
        # bar_amalg_plot_path = self.output_folder / f"{self.experiment_name}_amalgamated_snr_BAR.png"
        # fig_bar_amalg.savefig(bar_amalg_plot_path, dpi=300, bbox_inches='tight')
        # plt.close(fig_bar_amalg)
        # print(f"✓ Amalgamated SNR plot (bar): {bar_amalg_plot_path}")
        
        # 5. Heatmaps for key metrics
        # for metric in ['PESQ', 'SI_SDR', 'STOI']:
        #     fig_heat = self.plot_snr_noise_category_heatmap(metric=metric)
        #     if fig_heat is not None:
        #         heat_path = self.output_folder / f"{self.experiment_name}_heatmap_{metric}.png"
        #         fig_heat.savefig(heat_path, dpi=300, bbox_inches='tight')
        #         plt.close(fig_heat)
        #         print(f"✓ Heatmap ({metric}): {heat_path}")
        
        # 6. Individual SNR level plots and tables
        for snr in self.snr_levels:
            print(f"\n  Processing SNR {snr}dB...")
            
            # Bar chart for this SNR
            # fig_bar = self.plot_snr_bar_comparison(snr)
            # if fig_bar is not None:
            #     bar_path = self.output_folder / f"{self.experiment_name}_bar_{snr}dB.png"
            #     fig_bar.savefig(bar_path, dpi=300, bbox_inches='tight')
            #     plt.close(fig_bar)
            #     print(f"    ✓ Bar chart: {bar_path}")
            
            # Detailed data for this SNR
            if len(self.df_merged[snr]) > 0:
                detail_path = self.output_folder / f"{self.experiment_name}_detailed_{snr}dB.csv"
                self.df_merged[snr].to_csv(detail_path, index=False)
                print(f"    ✓ Detailed data: {detail_path}")
        
        print("\n" + "="*80)
        print(f"All results saved to: {self.output_folder}")
        print("="*80)


# Example usage
if __name__ == "__main__":
    # Define SNR levels to analyze
    snr_levels = [-5, 0, 5, 10, 15]
    
    # Define CSV file templates with {snr} placeholder
    csv_files_template = {
        'Noisy': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\BASELINE\NOIZEUS_EARS_BASELINE\BASELINE_NOIZEUS_EARS_[{snr}]dB.csv",
        # 'GTCRN': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\EXP3p1a\GTCRN_NOIZEUS_EARS_[{snr}]dB.csv",
        # 'GTCRN_SS': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\EXP3p1b_GTCRN_SS_pp_8ms_V0_f08\GTCRN_SS_TEST2_[{snr}]dB.csv",
        # 'GTCRN_SS_vad1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\EXP3p1b\EXP3p1b_GTCRN_SS_N4_lin_8ms_ov75_av1_nf1_f08_v1\GTCRN_SS_TEST2_[{snr}]dB.csv",
        # 'GTCRN_SS_delta': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\EXP3p1b\EXP3p1b_GTCRN_SS_delta15_N4_lin_8ms_ov75_av1_nf1_f08_v1\GTCRN_SS_TEST2_[{snr}]dB.csv",
        # 'WF_GTCRN_fr25_mu0.98_a_dd0.98_eta0.15': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\EXP3p1b_GTCRN_WF_ss\GTCRN_MWF_merged_[{snr}]dB.csv",
        # 'GTCRN_WF': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\GTCRN\GTCRNWF_EXP3p2a_25ms_quality\GTCRNWF_EXP3p2a_25ms_quality_merged_[{snr}]dB.csv",


        # Python transalation mband
        #'mband_py_lin_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1a_Python_mband_Test\mband_py_N6_lin_AVR0\mband_py_N6_lin_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_lin_avr0_future_frames': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1a_Python_mband_Test\mband_py_N6_lin_AVR0_8\mband_py_N6_lin_AVR0_8_[{snr}]dB_MERGED.csv",
        #'mband_py_lin_avr1_future_frames': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1a_Python_mband_Test\mband_py_N6_lin_AVR1_8\mband_py_N6_lin_AVR1_8_[{snr}]dB_MERGED.csv",
        #'mband_py_log_avr0_future_frames': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1a_Python_mband_Test\mband_py_N6_log_AVR0_8\mband_py_N6_log_AVR0_8_[{snr}]dB_MERGED.csv",
        #'mband_py_log_avr1_future_frames': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1a_Python_mband_Test\mband_py_N6_log_AVR1_8\mband_py_N6_log_AVR1_8_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_avr0_8': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1a_Python_mband_Test\mband_py_N6_mel_AVR0_8\mband_py_N6_mel_AVR0_8_[{snr}]dB_MERGED.csv",
        'mband_py_mel_avr1_future_frames': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1a_Python_mband_Test\mband_py_N6_mel_AVR1_8\mband_py_N6_mel_AVR1_8_[{snr}]dB_MERGED.csv",


        # Python mband causal AVRGING
        #'mband_py_lin_causal_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b_AVRGING_causal\mband_py_N6_lin\mband_py_N6_lin_[{snr}]dB_MERGED.csv",
        #'mband_py_log_causal_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b_AVRGING_causal\mband_py_N6_log\mband_py_N6_log_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_causal_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b_AVRGING_causal\mband_py_N6_mel\mband_py_N6_mel_[{snr}]dB_MERGED.csv",

        # Python mband causal AVRGING OG weights but ---XMAG----
        #'mband_py_lin_causal_ogw_avr1_xmag': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\AVRGING\SS_EXP1p1b_AVRGING_causal_ogweights\mband_py_N6_lin\mband_py_N6_lin_[{snr}]dB_MERGED.csv",
        #'mband_py_log_causal_ogw_avr1_xmag': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\AVRGING\SS_EXP1p1b_AVRGING_causal_ogweights\mband_py_N6_log\mband_py_N6_log_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_causal_ogw_avr1_xmag': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\AVRGING\SS_EXP1p1b_AVRGING_causal_ogweights\mband_py_N6_mel\mband_py_N6_mel_[{snr}]dB_MERGED.csv",

        # AVRGING CAUSAL AGAIN XMAGSM in mbans_AVG_causal.py
        # AVRGING =1 
        #'mband_py_lin_causal_avr1_og_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_log_causal_avr1_og_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_log_AVR1\mband_py_N6_log_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_causal_avr1_og_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_mel_AVR1\mband_py_N6_mel_AVR1_[{snr}]dB_MERGED.csv",

        #AVRGING =0 
        #'mband_py_lin_causal_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_lin_AVR0\mband_py_N6_lin_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_log_causal_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_log_AVR0\mband_py_N6_log_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_causal_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_mel_AVR0\mband_py_N6_mel_AVR0_[{snr}]dB_MERGED.csv",

        ## Conservative Weights  Wn2, Wn1, Wn0 = 0.12, 0.30, 0.58 
        #AVRGING = 1 
        #'mband_py_lin_conservative_wts_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_and_weight_change\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",

        ## IIR only 
        #AVRGING = 1 
        # 'mband_py_lin_IIR_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\IIR_only\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",

        # VAD 2 move for adaptive weights in avrging 
        #'mband_py_lin_VAD_adpt_wts_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\VAD2_move\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",

        # # Noiseupdt_stream  
        # 'mband_py_lin_noiseupdt_stream_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\VAD_stream\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",

        # Noiseupdt_stream with vad integrated before weighted smoothing   
        #'mband_py_lin_noiseupdt_b4_wtd_avg_stream_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\VAD_stream\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",

        # # Noiseupdt further stream with vad integration   
        # 'mband_py_lin_stream_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\VAD_stream_more\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",

        # Noiseupdt further stream with vad integration   
        #'mband_py_lin_stream_adpt_wts_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\VAD_stream_adapt_wts\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",

        # mband_stream.py
        # AVRGING CAUSAL AGAIN XMAGSM but in stream py file so can check for differences Conservative weights Wn2, Wn1, Wn0 = 0.12, 0.30, 0.58 
        # AVRGING =1 
        #'mband_py_lin_stream_avr1_cons_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream1\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",
        # 'mband_py_log_stream_avr1_cons_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream1\mband_py_N6_log_AVR1\mband_py_N6_log_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_stream_avr1_cons_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream1\mband_py_N6_mel_AVR1\mband_py_N6_mel_AVR1_[{snr}]dB_MERGED.csv",

        #AVRGING =0 
        # 'mband_py_lin_stream_avr0_cons_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream1\mband_py_N6_lin_AVR0\mband_py_N6_lin_AVR0_[{snr}]dB_MERGED.csv",
        # 'mband_py_log_stream_avr0_cons_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream1\mband_py_N6_log_AVR0\mband_py_N6_log_AVR0_[{snr}]dB_MERGED.csv",
        # 'mband_py_mel_stream_avr0_cons_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream1\mband_py_N6_mel_AVR0\mband_py_N6_mel_AVR0_[{snr}]dB_MERGED.csv",

        # XMAGSM in stream with OG weights Wn2, Wn1, Wn0 = 0.09, 0.25, 0.66
        # Not actually streaming but testing results, in mband_stream.py
        # AVRGING =1 
        #'mband_py_lin_in_stream_avr1_og_wts_xmagsm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_og_wts\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_log_in_stream_avr1_og_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_og_wts\mband_py_N6_log_AVR1\mband_py_N6_log_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_in_stream_avr1_og_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_og_wts\mband_py_N6_mel_AVR1\mband_py_N6_mel_AVR1_[{snr}]dB_MERGED.csv",

        # AVRGING =0 
        #'mband_py_lin_xmagsm_avr0_og_wts': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_og_wts\mband_py_N6_lin_AVR0\mband_py_N6_lin_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_log_stream_avr0_og_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_og_wts\mband_py_N6_log_AVR0\mband_py_N6_log_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_stream_avr0_og_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_og_wts\mband_py_N6_mel_AVR0\mband_py_N6_mel_AVR0_[{snr}]dB_MERGED.csv",

        # XMAGSM in stream with circular buffer and adaptive weights mband_stream.py below -3dB
        # AVRGING =1 
        #'mband_py_lin_circular_stream_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_log_circular_stream_adapt_wts_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer\mband_py_N6_log_AVR1\mband_py_N6_log_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_circular_stream_adapt_wts_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer\mband_py_N6_mel_AVR1\mband_py_N6_mel_AVR1_[{snr}]dB_MERGED.csv",

        # AVRGING =0 
        #'mband_py_lin_circular_stream_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer\mband_py_N6_lin_AVR0\mband_py_N6_lin_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_log_circular_stream_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer\mband_py_N6_log_AVR0\mband_py_N6_log_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_circular_stream_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer\mband_py_N6_mel_AVR0\mband_py_N6_mel_AVR0_[{snr}]dB_MERGED.csv",

        # Say which function it is from 
        # From mband_full_stream_og_wts.py Testing full stream with original weights
        # Note full and circular stream are the same
        # AVRGING =1 
        #'mband_py_lin_full_stream_og_wts_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_og_wts\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_log_full_stream_og_wts_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_og_wts\mband_py_N6_log_AVR1\mband_py_N6_log_AVR1_[{snr}]dB_MERGED.csv",
        'mband_py_mel_full_stream_og_wts_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_og_wts\mband_py_N6_mel_AVR1\mband_py_N6_mel_AVR1_[{snr}]dB_MERGED.csv",

        # AVRGING =0 
        #'mband_py_lin_circular_stream_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_og_wts\mband_py_N6_lin_AVR0\mband_py_N6_lin_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_log_full_stream_og_wts_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_og_wts\mband_py_N6_log_AVR0\mband_py_N6_log_AVR0_[{snr}]dB_MERGED.csv",
        'mband_py_mel_full_stream_og_wts_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_og_wts\mband_py_N6_mel_AVR0\mband_py_N6_mel_AVR0_[{snr}]dB_MERGED.csv",

        # From mband_full_stream_VAD_below.py moved VAD below avrging so acceots a smoother frames for noise updates
        # AVRGING =1 
        #'mband_py_lin_full_stream_VAD_below_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_vad_below\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_log_full_stream_VAD_below_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_vad_below\mband_py_N6_log_AVR1\mband_py_N6_log_AVR1_[{snr}]dB_MERGED.csv",
        'mband_py_mel_full_stream_VAD_below_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_vad_below\mband_py_N6_mel_AVR1\mband_py_N6_mel_AVR1_[{snr}]dB_MERGED.csv",

        # AVRGING =0 
        #'mband_py_lin_full_stream_VAD_below_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_vad_below\mband_py_N6_lin_AVR0\mband_py_N6_lin_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_log_full_stream_VAD_below_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_vad_below\mband_py_N6_log_AVR0\mband_py_N6_log_AVR0_[{snr}]dB_MERGED.csv",
        'mband_py_mel_full_stream_VAD_below_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\stream\stream_circ_buffer_vad_below\mband_py_N6_mel_AVR0\mband_py_N6_mel_AVR0_[{snr}]dB_MERGED.csv",

        # Also say where saved? 

    }
    
    # Set output folder
    output_folder = r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\compare_csvs\EXP1\spectral\stream\vad_below_AVRGING\mel_avr10"
    
    # Create comparator
    comparator = MultiSNRAudioEnhancementComparator(
        csv_files_template,
        snr_levels=snr_levels,
        output_folder=output_folder,
        experiment_name="Multi_SNR"
    )
    
    # Export all results
    comparator.export_all_results()
    
    print("\n✓ Multi-SNR analysis complete! Check the output folder for:")
    print("  - Amalgamated comparison across all SNR levels")
    print("  - Individual bar charts for each SNR level")
    print("  - Heatmaps showing SNR vs Noise Category performance")
    print("  - Detailed CSV files for further analysis")


#### Past Files #####

#===========================================================
##### MATLAB ######


        #'specsub': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP_MATLAB_COMPARE\specsub\specsub_[{snr}]dB_MERGED.csv",
        #'ss_rdc': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP_MATLAB_COMPARE\ss_rdc\ss_rdc_[{snr}]dB_MERGED.csv",
        #'ss_rdc_og': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP_MATLAB_COMPARE\ss_rdc_og\ss_rdc_og_[{snr}]dB_MERGED.csv",
        
        # 'mband_lin_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP_MATLAB_COMPARE\mband_N6_lin_AVR0\mband_N6_lin_AVR0_[{snr}]dB_MERGED.csv",
        # 'mband_lin_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP_MATLAB_COMPARE\mband_N6_lin_AVR1\mband_N6_lin_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_log_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP_MATLAB_COMPARE\mband_N6_log_AVR0\mband_N6_log_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_log_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP_MATLAB_COMPARE\mband_N6_log_AVR1\mband_N6_log_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_mel_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP_MATLAB_COMPARE\mband_N6_mel_AVR0\mband_N6_mel_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_mel_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP_MATLAB_COMPARE\mband_N6_mel_AVR1\mband_N6_mel_AVR1_[{snr}]dB_MERGED.csv",

#===========================================================
        # DELTAS
        # AVRGING = 1
        #'mband_py_lin_deltas_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b_deltas\mband_py_N6_lin\mband_py_N6_lin_[{snr}]dB_MERGED.csv",
        #'mband_py_log_deltas_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b_deltas\mband_py_N6_log\mband_py_N6_log_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_deltas': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b_deltas\mband_py_N6_mel\mband_py_N6_mel_[{snr}]dB_MERGED.csv",

        # AVRGING = 0
        #'mband_py_lin_deltas_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b_deltas\mband_py_N6_lin_AVR0\mband_py_N6_lin_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_log_deltas_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b_deltas\mband_py_N6_log_AVR0\mband_py_N6_log_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_deltas_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b_deltas\mband_py_N6_mel_AVR0\mband_py_N6_mel_AVR0_[{snr}]dB_MERGED.csv",

        #===========================================================

        
        # AVRGING CAUSAL AGAIN XMAGSM
        # AVRGING =1 
        #'mband_py_lin_causal_avr1_og_wts_xmgasm': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_lin_AVR1\mband_py_N6_lin_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_log_causal_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_log_AVR1\mband_py_N6_log_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_causal_avr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_mel_AVR1\mband_py_N6_mel_AVR1_[{snr}]dB_MERGED.csv",

        #AVRGING =0 
        #'mband_py_lin_causal_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_lin_AVR0\mband_py_N6_lin_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_log_causal_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_log_AVR0\mband_py_N6_log_AVR0_[{snr}]dB_MERGED.csv",
        #'mband_py_mel_causal_avr0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\SS_EXP1p1b\AVRGING\xmagsm_change\mband_py_N6_mel_AVR0\mband_py_N6_mel_AVR0_[{snr}]dB_MERGED.csv",