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
        if len(self.method_names) > 8:
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
        
        # print(f"  Merged dataset for {snr}dB: {len(df_merged)} common files")
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
            """Categorize each NOIZEUS noise file explicitly."""
            filename_lower = str(filename).lower()

            if 'cafeteria_babble' in filename_lower:
                return 'Cafeteria Babble'
            elif 'car noise_60mph' in filename_lower:
                return 'Car Noise 60mph'
            elif 'car noise_idle noise_40mph' in filename_lower:
                return 'Car Noise Idle 40mph'
            elif 'car noise_idle noise_60mph' in filename_lower:
                return 'Car Noise Idle 60mph'
            elif 'construction_crane_moving' in filename_lower:
                return 'Construction Crane Moving'
            elif 'construction_drilling' in filename_lower:
                return 'Construction Drilling'
            elif 'construction_jackhammer1' in filename_lower:
                return 'Construction Jackhammer 1'
            elif 'construction_jackhammer2' in filename_lower:
                return 'Construction Jackhammer 2'
            elif 'construction_trucks_unloading' in filename_lower:
                return 'Construction Trucks Unloading'
            elif 'inside flight' in filename_lower:
                return 'Inside Flight'
            elif 'inside train_1' in filename_lower:
                return 'Inside Train 1'
            elif 'inside train_2' in filename_lower:
                return 'Inside Train 2'
            elif 'inside train_3' in filename_lower:
                return 'Inside Train 3'
            elif 'pc fan noise' in filename_lower:
                return 'PC Fan Noise'
            elif 'ssn_ieee' in filename_lower:
                return 'SSN IEEE'
            elif 'street noise_downtown' in filename_lower:
                return 'Street Noise Downtown'
            elif 'street noise' in filename_lower:
                return 'Street Noise'
            elif 'train1' in filename_lower:
                return 'Train 1'
            elif 'train2' in filename_lower:
                return 'Train 2'
            elif 'water cooler' in filename_lower:
                return 'Water Cooler'
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
                    
                    # # Add value labels
                    # for snr, val in zip(snr_values, metric_values):
                    #     ax.annotate(f'{val:.3f}', 
                    #                xy=(snr, val), 
                    #                xytext=(0, 8),
                    #                textcoords='offset points',
                    #                ha='center', va='bottom',
                    #                fontsize=14, fontweight='bold')
            
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

    def plot_snr_noise_category_heatmap(self, metric='PESQ', figsize=(16, 10), highlight_best=True):
        """
        Create heatmap showing performance by SNR and noise category.
        Separate heatmap for each method with best performer highlighted.
        
        Parameters:
        -----------
        highlight_best : bool
            If True, circles the best performing method for each (category, SNR) combination
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
        
        # Collect all data matrices to find best performers
        all_data_matrices = {}
        
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
            
            all_data_matrices[method_name] = data_matrix
            
            # Plot heatmap with metric-appropriate range
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(self.snr_levels)))
            ax.set_xticklabels([f'{snr}' for snr in self.snr_levels], fontsize=16)
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
            ax.set_xlabel('SNR Level (dB)', fontsize=16)
            
            if idx == 0:
                ax.set_ylabel('Noise Category', fontsize=16)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Highlight best performers if requested
        if highlight_best:
            for i in range(len(categories)):
                for j in range(len(self.snr_levels)):
                    # Collect values from all methods for this cell
                    values = []
                    for method_name in self.method_names:
                        val = all_data_matrices[method_name][i, j]
                        if not np.isnan(val):
                            values.append((val, method_name))
                    
                    if values:
                        # Find best value (higher is better for all metrics)
                        best_val, best_method = max(values, key=lambda x: x[0])
                        best_idx = self.method_names.index(best_method)
                        
                        # Draw circle around best performer
                        circle = plt.Circle((j, i), 0.50, color='blue', fill=False, 
                                          linewidth=0.7, transform=axes[best_idx].transData)
                        axes[best_idx].add_patch(circle)
        
        plt.suptitle(f'{metric} Performance: SNR vs Noise Category (Best Performers Circled)', 
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

    def analyze_best_performers(self):
        """
        Comprehensive analysis of which algorithm performs best under different conditions.
        Returns detailed statistics about best performers.
        """
        analysis = {
            'overall': {},
            'by_metric': {},
            'by_snr': {},
            'by_noise': {},
            'win_matrix': {}
        }
        
        # For each metric, count wins per method
        for metric in self.metrics:
            win_counts = {method: 0 for method in self.method_names}
            total_comparisons = 0
            
            for snr in self.snr_levels:
                df = self.df_merged[snr]
                if len(df) == 0:
                    continue
                
                # Check each file
                for idx, row in df.iterrows():
                    values = []
                    for method in self.method_names:
                        col = f'{metric}_{method}'
                        if col in df.columns and not pd.isna(row[col]):
                            values.append((row[col], method))
                    
                    if values:
                        best_val, best_method = max(values, key=lambda x: x[0])
                        win_counts[best_method] += 1
                        total_comparisons += 1
            
            analysis['by_metric'][metric] = {
                'win_counts': win_counts,
                'win_percentages': {m: (c/total_comparisons*100 if total_comparisons > 0 else 0) 
                                for m, c in win_counts.items()},
                'total_comparisons': total_comparisons
            }
        
        # Win counts by SNR level
        for snr in self.snr_levels:
            df = self.df_merged[snr]
            if len(df) == 0:
                continue
            
            snr_wins = {method: 0 for method in self.method_names}
            snr_total = 0
            
            for metric in self.metrics:
                for idx, row in df.iterrows():
                    values = []
                    for method in self.method_names:
                        col = f'{metric}_{method}'
                        if col in df.columns and not pd.isna(row[col]):
                            values.append((row[col], method))
                    
                    if values:
                        best_val, best_method = max(values, key=lambda x: x[0])
                        snr_wins[best_method] += 1
                        snr_total += 1
            
            analysis['by_snr'][snr] = {
                'win_counts': snr_wins,
                'win_percentages': {m: (c/snr_total*100 if snr_total > 0 else 0) 
                                for m, c in snr_wins.items()}
            }
        
        # Win counts by noise category
        all_categories = set()
        for snr in self.snr_levels:
            if 'noise_category' in self.df_merged[snr].columns:
                all_categories.update(self.df_merged[snr]['noise_category'].unique())
        
        for category in all_categories:
            cat_wins = {method: 0 for method in self.method_names}
            cat_total = 0
            
            for snr in self.snr_levels:
                df = self.df_merged[snr]
                if len(df) == 0 or 'noise_category' not in df.columns:
                    continue
                
                cat_df = df[df['noise_category'] == category]
                
                for metric in self.metrics:
                    for idx, row in cat_df.iterrows():
                        values = []
                        for method in self.method_names:
                            col = f'{metric}_{method}'
                            if col in cat_df.columns and not pd.isna(row[col]):
                                values.append((row[col], method))
                        
                        if values:
                            best_val, best_method = max(values, key=lambda x: x[0])
                            cat_wins[best_method] += 1
                            cat_total += 1
            
            if cat_total > 0:
                analysis['by_noise'][category] = {
                    'win_counts': cat_wins,
                    'win_percentages': {m: (c/cat_total*100) for m, c in cat_wins.items()},
                    'total_comparisons': cat_total
                }
        
        # Overall win counts
        overall_wins = {method: 0 for method in self.method_names}
        overall_total = 0
        
        for snr in self.snr_levels:
            df = self.df_merged[snr]
            if len(df) == 0:
                continue
            
            for metric in self.metrics:
                for idx, row in df.iterrows():
                    values = []
                    for method in self.method_names:
                        col = f'{metric}_{method}'
                        if col in df.columns and not pd.isna(row[col]):
                            values.append((row[col], method))
                    
                    if values:
                        best_val, best_method = max(values, key=lambda x: x[0])
                        overall_wins[best_method] += 1
                        overall_total += 1
        
        analysis['overall'] = {
            'win_counts': overall_wins,
            'win_percentages': {m: (c/overall_total*100 if overall_total > 0 else 0) 
                            for m, c in overall_wins.items()},
            'total_comparisons': overall_total
        }
        
        return analysis

    def generate_critical_analysis_report(self):
        """
        Generate a critical analysis report identifying best performers and patterns.
        """
        analysis = self.analyze_best_performers()
        
        report_lines = []
        report_lines.append("\n" + "="*120)
        report_lines.append("CRITICAL ANALYSIS: BEST PERFORMER IDENTIFICATION")
        report_lines.append("="*120)
        
        # Overall winner
        report_lines.append("\n>>> OVERALL PERFORMANCE <<<")
        overall = analysis['overall']
        sorted_overall = sorted(overall['win_counts'].items(), key=lambda x: x[1], reverse=True)
        
        report_lines.append(f"\nTotal comparisons: {overall['total_comparisons']}")
        report_lines.append("\nRanking (by number of wins across all metrics, SNRs, and files):")
        for rank, (method, wins) in enumerate(sorted_overall, 1):
            pct = overall['win_percentages'][method]
            report_lines.append(f"  {rank}. {method:20s}: {wins:5d} wins ({pct:5.1f}%)")
        
        # Best by metric
        report_lines.append("\n" + "-"*120)
        report_lines.append("\n>>> BEST PERFORMER BY METRIC <<<")
        
        for metric in self.metrics:
            metric_data = analysis['by_metric'][metric]
            best_method = max(metric_data['win_counts'].items(), key=lambda x: x[1])[0]
            best_pct = metric_data['win_percentages'][best_method]
            
            report_lines.append(f"\n{metric}:")
            report_lines.append(f"  Winner: {best_method} ({best_pct:.1f}% of {metric_data['total_comparisons']} comparisons)")
            
            sorted_methods = sorted(metric_data['win_counts'].items(), key=lambda x: x[1], reverse=True)
            for method, wins in sorted_methods:
                pct = metric_data['win_percentages'][method]
                report_lines.append(f"    {method:20s}: {wins:4d} wins ({pct:5.1f}%)")
        
        # Best by SNR level
        report_lines.append("\n" + "-"*120)
        report_lines.append("\n>>> BEST PERFORMER BY SNR LEVEL <<<")
        
        for snr in sorted(analysis['by_snr'].keys()):
            snr_data = analysis['by_snr'][snr]
            best_method = max(snr_data['win_counts'].items(), key=lambda x: x[1])[0]
            best_pct = snr_data['win_percentages'][best_method]
            
            report_lines.append(f"\nSNR {snr}dB:")
            report_lines.append(f"  Winner: {best_method} ({best_pct:.1f}%)")
            
            sorted_methods = sorted(snr_data['win_counts'].items(), key=lambda x: x[1], reverse=True)
            for method, wins in sorted_methods:
                pct = snr_data['win_percentages'][method]
                report_lines.append(f"    {method:20s}: {wins:4d} wins ({pct:5.1f}%)")
        
        # Best by noise category
        if analysis['by_noise']:
            report_lines.append("\n" + "-"*120)
            report_lines.append("\n>>> BEST PERFORMER BY NOISE CATEGORY <<<")
            
            for category in sorted(analysis['by_noise'].keys()):
                cat_data = analysis['by_noise'][category]
                best_method = max(cat_data['win_counts'].items(), key=lambda x: x[1])[0]
                best_pct = cat_data['win_percentages'][best_method]
                
                report_lines.append(f"\n{category}:")
                report_lines.append(f"  Winner: {best_method} ({best_pct:.1f}% of {cat_data['total_comparisons']} comparisons)")
                
                sorted_methods = sorted(cat_data['win_counts'].items(), key=lambda x: x[1], reverse=True)
                for method, wins in sorted_methods[:3]:  # Show top 3
                    pct = cat_data['win_percentages'][method]
                    report_lines.append(f"    {method:20s}: {wins:4d} wins ({pct:5.1f}%)")
        
        # Key insights
        report_lines.append("\n" + "="*120)
        report_lines.append("\n>>> KEY INSIGHTS <<<")
        
        # Which method is most consistent (wins across different conditions)?
        consistency_scores = {}
        for method in self.method_names:
            wins_by_metric = sum(1 for m in analysis['by_metric'].values() 
                            if max(m['win_counts'].items(), key=lambda x: x[1])[0] == method)
            wins_by_snr = sum(1 for s in analysis['by_snr'].values() 
                            if max(s['win_counts'].items(), key=lambda x: x[1])[0] == method)
            consistency_scores[method] = wins_by_metric + wins_by_snr
        
        most_consistent = max(consistency_scores.items(), key=lambda x: x[1])
        report_lines.append(f"\nMost Consistent Performer: {most_consistent[0]}")
        report_lines.append(f"  (Best performer in {most_consistent[1]} different metric/SNR combinations)")
        
        # Identify strengths and weaknesses
        report_lines.append("\n\nAlgorithm Specializations:")
        for method in self.method_names:
            strengths = []
            
            # Check metric strengths
            for metric, data in analysis['by_metric'].items():
                if max(data['win_counts'].items(), key=lambda x: x[1])[0] == method:
                    strengths.append(f"Best at {metric}")
            
            # Check SNR strengths
            strong_snrs = [snr for snr, data in analysis['by_snr'].items() 
                        if max(data['win_counts'].items(), key=lambda x: x[1])[0] == method]
            if strong_snrs:
                strengths.append(f"Dominates at {', '.join(map(str, strong_snrs))}dB")
            
            if strengths:
                report_lines.append(f"\n{method}:")
                for strength in strengths:
                    report_lines.append(f"  • {strength}")
        
        report_lines.append("\n" + "="*120)
        
        return "\n".join(report_lines)

    def analyze_hybrid_improvements(self, baseline_method='GTCRN'):
        """
        Analyze how hybrid methods (GTCRN-SS, GTCRN-WF) improve over baseline GTCRN.
        Identifies best use cases for each hybrid approach.
        
        Parameters:
        -----------
        baseline_method : str
            The baseline method to compare against (default: 'GTCRN')
        """
        if baseline_method not in self.method_names:
            print(f"ERROR: Baseline method '{baseline_method}' not found in methods")
            return None
        
        # Identify hybrid methods (methods containing the baseline name but different)
        hybrid_methods = [m for m in self.method_names 
                        if baseline_method in m and m != baseline_method]
        
        if not hybrid_methods:
            print(f"No hybrid methods found containing '{baseline_method}'")
            return None
        
        analysis = {
            'overall': {},
            'by_metric': {},
            'by_snr': {},
            'by_noise': {},
            'degradation_analysis': {}
        }
        
        # For each hybrid method
        for hybrid in hybrid_methods:
            analysis['overall'][hybrid] = {}
            analysis['by_metric'][hybrid] = {}
            analysis['by_snr'][hybrid] = {}
            analysis['by_noise'][hybrid] = {}
            analysis['degradation_analysis'][hybrid] = {}
            
            # Overall improvement statistics
            total_improvements = 0
            total_degradations = 0
            total_comparisons = 0
            improvement_sum = 0
            
            for snr in self.snr_levels:
                df = self.df_merged[snr]
                if len(df) == 0:
                    continue
                
                for metric in self.metrics:
                    baseline_col = f'{metric}_{baseline_method}'
                    hybrid_col = f'{metric}_{hybrid}'
                    
                    if baseline_col not in df.columns or hybrid_col not in df.columns:
                        continue
                    
                    differences = df[hybrid_col] - df[baseline_col]
                    improvements = (differences > 0).sum()
                    degradations = (differences < 0).sum()
                    
                    total_improvements += improvements
                    total_degradations += degradations
                    total_comparisons += len(differences)
                    improvement_sum += differences.sum()
            
            analysis['overall'][hybrid] = {
                'improvements': total_improvements,
                'degradations': total_degradations,
                'unchanged': total_comparisons - total_improvements - total_degradations,
                'total': total_comparisons,
                'improvement_rate': (total_improvements / total_comparisons * 100) if total_comparisons > 0 else 0,
                'degradation_rate': (total_degradations / total_comparisons * 100) if total_comparisons > 0 else 0,
                'avg_improvement': (improvement_sum / total_comparisons) if total_comparisons > 0 else 0
            }
            
            # Improvement by metric
            for metric in self.metrics:
                metric_improvements = 0
                metric_degradations = 0
                metric_total = 0
                metric_improvement_sum = 0
                
                for snr in self.snr_levels:
                    df = self.df_merged[snr]
                    if len(df) == 0:
                        continue
                    
                    baseline_col = f'{metric}_{baseline_method}'
                    hybrid_col = f'{metric}_{hybrid}'
                    
                    if baseline_col not in df.columns or hybrid_col not in df.columns:
                        continue
                    
                    differences = df[hybrid_col] - df[baseline_col]
                    metric_improvements += (differences > 0).sum()
                    metric_degradations += (differences < 0).sum()
                    metric_total += len(differences)
                    metric_improvement_sum += differences.sum()
                
                if metric_total > 0:
                    analysis['by_metric'][hybrid][metric] = {
                        'improvements': metric_improvements,
                        'degradations': metric_degradations,
                        'improvement_rate': (metric_improvements / metric_total * 100),
                        'degradation_rate': (metric_degradations / metric_total * 100),
                        'avg_improvement': (metric_improvement_sum / metric_total)
                    }
            
            # Improvement by SNR
            for snr in self.snr_levels:
                df = self.df_merged[snr]
                if len(df) == 0:
                    continue
                
                snr_improvements = 0
                snr_degradations = 0
                snr_total = 0
                snr_improvement_sum = 0
                
                for metric in self.metrics:
                    baseline_col = f'{metric}_{baseline_method}'
                    hybrid_col = f'{metric}_{hybrid}'
                    
                    if baseline_col not in df.columns or hybrid_col not in df.columns:
                        continue
                    
                    differences = df[hybrid_col] - df[baseline_col]
                    snr_improvements += (differences > 0).sum()
                    snr_degradations += (differences < 0).sum()
                    snr_total += len(differences)
                    snr_improvement_sum += differences.sum()
                
                if snr_total > 0:
                    analysis['by_snr'][hybrid][snr] = {
                        'improvements': snr_improvements,
                        'degradations': snr_degradations,
                        'improvement_rate': (snr_improvements / snr_total * 100),
                        'avg_improvement': (snr_improvement_sum / snr_total)
                    }
            
            # Improvement by noise category
            all_categories = set()
            for snr in self.snr_levels:
                if 'noise_category' in self.df_merged[snr].columns:
                    all_categories.update(self.df_merged[snr]['noise_category'].unique())
            
            for category in all_categories:
                cat_improvements = 0
                cat_degradations = 0
                cat_total = 0
                cat_improvement_sum = 0
                
                for snr in self.snr_levels:
                    df = self.df_merged[snr]
                    if len(df) == 0 or 'noise_category' not in df.columns:
                        continue
                    
                    cat_df = df[df['noise_category'] == category]
                    
                    for metric in self.metrics:
                        baseline_col = f'{metric}_{baseline_method}'
                        hybrid_col = f'{metric}_{hybrid}'
                        
                        if baseline_col not in cat_df.columns or hybrid_col not in cat_df.columns:
                            continue
                        
                        differences = cat_df[hybrid_col] - cat_df[baseline_col]
                        cat_improvements += (differences > 0).sum()
                        cat_degradations += (differences < 0).sum()
                        cat_total += len(differences)
                        cat_improvement_sum += differences.sum()
                
                if cat_total > 0:
                    analysis['by_noise'][hybrid][category] = {
                        'improvements': cat_improvements,
                        'degradations': cat_degradations,
                        'improvement_rate': (cat_improvements / cat_total * 100),
                        'avg_improvement': (cat_improvement_sum / cat_total)
                    }
        
        return analysis

    def generate_hybrid_analysis_report(self, baseline_method='GTCRN'):
        """
        Generate comprehensive report on hybrid method improvements.
        """
        analysis = self.analyze_hybrid_improvements(baseline_method)
        
        if analysis is None:
            return "Unable to generate hybrid analysis report."
        
        report_lines = []
        report_lines.append("\n" + "="*120)
        report_lines.append(f"HYBRID METHOD ANALYSIS: Improvements over {baseline_method}")
        report_lines.append("="*120)
        
        hybrid_methods = list(analysis['overall'].keys())
        
        # Overall comparison
        report_lines.append("\n>>> OVERALL IMPROVEMENT SUMMARY <<<\n")
        for hybrid in hybrid_methods:
            data = analysis['overall'][hybrid]
            report_lines.append(f"{hybrid}:")
            report_lines.append(f"  Total Comparisons:   {data['total']:5d}")
            report_lines.append(f"  Improvements:        {data['improvements']:5d} ({data['improvement_rate']:5.1f}%)")
            report_lines.append(f"  Degradations:        {data['degradations']:5d} ({data['degradation_rate']:5.1f}%)")
            report_lines.append(f"  Unchanged:           {data['unchanged']:5d}")
            report_lines.append(f"  Average Improvement: {data['avg_improvement']:+.4f}")
            report_lines.append("")
        
        # Best hybrid method overall
        best_hybrid = max(hybrid_methods, 
                        key=lambda h: analysis['overall'][h]['improvement_rate'])
        report_lines.append(f"🏆 WINNER: {best_hybrid} has highest improvement rate "
                        f"({analysis['overall'][best_hybrid]['improvement_rate']:.1f}%)\n")
        
        # Improvement by metric
        report_lines.append("\n" + "-"*120)
        report_lines.append("\n>>> IMPROVEMENT BY METRIC <<<\n")
        
        for metric in self.metrics:
            report_lines.append(f"{metric}:")
            for hybrid in hybrid_methods:
                if metric in analysis['by_metric'][hybrid]:
                    data = analysis['by_metric'][hybrid][metric]
                    report_lines.append(f"  {hybrid}:")
                    report_lines.append(f"    Improvement Rate: {data['improvement_rate']:5.1f}%")
                    report_lines.append(f"    Avg Improvement:  {data['avg_improvement']:+.4f}")
            
            # Best for this metric
            best_for_metric = max(hybrid_methods,
                                key=lambda h: analysis['by_metric'][h].get(metric, {}).get('improvement_rate', 0))
            best_rate = analysis['by_metric'][best_for_metric].get(metric, {}).get('improvement_rate', 0)
            report_lines.append(f"  → Best: {best_for_metric} ({best_rate:.1f}% improvement rate)\n")
        
        # Improvement by SNR
        report_lines.append("\n" + "-"*120)
        report_lines.append("\n>>> IMPROVEMENT BY SNR LEVEL <<<\n")
        
        for snr in sorted(self.snr_levels):
            report_lines.append(f"SNR {snr}dB:")
            for hybrid in hybrid_methods:
                if snr in analysis['by_snr'][hybrid]:
                    data = analysis['by_snr'][hybrid][snr]
                    report_lines.append(f"  {hybrid}:")
                    report_lines.append(f"    Improvement Rate: {data['improvement_rate']:5.1f}%")
                    report_lines.append(f"    Avg Improvement:  {data['avg_improvement']:+.4f}")
            
            # Best for this SNR
            best_for_snr = max(hybrid_methods,
                            key=lambda h: analysis['by_snr'][h].get(snr, {}).get('improvement_rate', 0))
            best_rate = analysis['by_snr'][best_for_snr].get(snr, {}).get('improvement_rate', 0)
            report_lines.append(f"  → Best: {best_for_snr} ({best_rate:.1f}% improvement rate)\n")
        
        # Improvement by noise category
        report_lines.append("\n" + "-"*120)
        report_lines.append("\n>>> IMPROVEMENT BY NOISE CATEGORY <<<")
        report_lines.append("(Top improvement rates shown)\n")
        
        all_categories = set()
        for hybrid in hybrid_methods:
            all_categories.update(analysis['by_noise'][hybrid].keys())
        
        for category in sorted(all_categories):
            report_lines.append(f"\n{category}:")
            
            category_results = []
            for hybrid in hybrid_methods:
                if category in analysis['by_noise'][hybrid]:
                    data = analysis['by_noise'][hybrid][category]
                    category_results.append((hybrid, data['improvement_rate'], data['avg_improvement']))
            
            # Sort by improvement rate
            category_results.sort(key=lambda x: x[1], reverse=True)
            
            for hybrid, imp_rate, avg_imp in category_results:
                report_lines.append(f"  {hybrid:20s}: {imp_rate:5.1f}% improvement rate, "
                                f"avg {avg_imp:+.4f}")
            
            if category_results:
                report_lines.append(f"  → Best: {category_results[0][0]}")
        
        # KEY RECOMMENDATIONS
        report_lines.append("\n" + "="*120)
        report_lines.append("\n>>> KEY RECOMMENDATIONS <<<\n")
        
        # Find where each hybrid excels
        for hybrid in hybrid_methods:
            report_lines.append(f"{hybrid} EXCELS AT:")
            
            # Best metrics
            best_metrics = []
            for metric in self.metrics:
                if metric in analysis['by_metric'][hybrid]:
                    imp_rate = analysis['by_metric'][hybrid][metric]['improvement_rate']
                    if imp_rate > 50:  # Threshold for "excels"
                        best_metrics.append((metric, imp_rate))
            
            if best_metrics:
                best_metrics.sort(key=lambda x: x[1], reverse=True)
                for metric, rate in best_metrics[:3]:  # Top 3
                    report_lines.append(f"  ✓ {metric}: {rate:.1f}% improvement rate")
            
            # Best SNR ranges
            best_snrs = []
            for snr in self.snr_levels:
                if snr in analysis['by_snr'][hybrid]:
                    imp_rate = analysis['by_snr'][hybrid][snr]['improvement_rate']
                    if imp_rate > 50:
                        best_snrs.append((snr, imp_rate))
            
            if best_snrs:
                best_snrs.sort(key=lambda x: x[1], reverse=True)
                snr_list = [f"{snr}dB" for snr, _ in best_snrs]
                report_lines.append(f"  ✓ SNR levels: {', '.join(snr_list)}")
            
            # Best noise types
            best_noises = []
            for category, data in analysis['by_noise'][hybrid].items():
                if data['improvement_rate'] > 60:
                    best_noises.append((category, data['improvement_rate']))
            
            if best_noises:
                best_noises.sort(key=lambda x: x[1], reverse=True)
                report_lines.append(f"  ✓ Noise types:")
                for noise, rate in best_noises[:5]:  # Top 5
                    report_lines.append(f"      • {noise}: {rate:.1f}%")
            
            report_lines.append("")
        
        # When to use each
        report_lines.append("\n" + "-"*120)
        report_lines.append("\n>>> USAGE GUIDELINES <<<\n")
        
        # Compare hybrids directly
        if len(hybrid_methods) >= 2:
            h1, h2 = hybrid_methods[0], hybrid_methods[1]
            
            h1_better_metrics = []
            h2_better_metrics = []
            
            for metric in self.metrics:
                if metric in analysis['by_metric'][h1] and metric in analysis['by_metric'][h2]:
                    rate1 = analysis['by_metric'][h1][metric]['improvement_rate']
                    rate2 = analysis['by_metric'][h2][metric]['improvement_rate']
                    
                    if rate1 > rate2:
                        h1_better_metrics.append(metric)
                    else:
                        h2_better_metrics.append(metric)
            
            report_lines.append(f"Choose {h1} when prioritizing:")
            for metric in h1_better_metrics:
                report_lines.append(f"  • {metric}")
            
            report_lines.append(f"\nChoose {h2} when prioritizing:")
            for metric in h2_better_metrics:
                report_lines.append(f"  • {metric}")
        
        report_lines.append("\n" + "="*120)
        
        return "\n".join(report_lines)
    
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
        fig_bar_amalg = self.plot_amalgamated_snr_bar_comparison()
        bar_amalg_plot_path = self.output_folder / f"{self.experiment_name}_amalgamated_snr_BAR.png"
        fig_bar_amalg.savefig(bar_amalg_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig_bar_amalg)
        print(f"✓ Amalgamated SNR plot (bar): {bar_amalg_plot_path}")
        
        # 5. Heatmaps for key metrics
        for metric in ['PESQ', 'SI_SDR', 'STOI', 'DNSMOS_p808_mos', 'DNSMOS_mos_sig', 'DNSMOS_mos_bak', 'DNSMOS_mos_ovr']:
            fig_heat = self.plot_snr_noise_category_heatmap(metric=metric)
            if fig_heat is not None:
                heat_path = self.output_folder / f"{self.experiment_name}_heatmap_{metric}.png"
                fig_heat.savefig(heat_path, dpi=300, bbox_inches='tight')
                plt.close(fig_heat)
                print(f"✓ Heatmap ({metric}): {heat_path}")
        
        # 6. Individual SNR level plots and tables
        for snr in self.snr_levels:
            print(f"\n  Processing SNR {snr}dB...")
            
            #Bar chart for this SNR
            fig_bar = self.plot_snr_bar_comparison(snr)
            if fig_bar is not None:
                bar_path = self.output_folder / f"{self.experiment_name}_bar_{snr}dB.png"
                fig_bar.savefig(bar_path, dpi=300, bbox_inches='tight')
                plt.close(fig_bar)
                print(f"    ✓ Bar chart: {bar_path}")
            
            # Detailed data for this SNR
            if len(self.df_merged[snr]) > 0:
                detail_path = self.output_folder / f"{self.experiment_name}_detailed_{snr}dB.csv"
                self.df_merged[snr].to_csv(detail_path, index=False)
                print(f"    ✓ Detailed data: {detail_path}")

        # analyze best performers
        critical_report = self.generate_critical_analysis_report()
        critical_path = self.output_folder / f"{self.experiment_name}_CRITICAL_ANALYSIS.txt"
        with open(critical_path, 'w') as f:
            f.write(critical_report)
        print(f"\n✓ Critical analysis report: {critical_path}")

        # analyze hybrid improvements
        hybrid_report = self.generate_hybrid_analysis_report(baseline_method='GTCRN')
        hybrid_path = self.output_folder / f"{self.experiment_name}_HYBRID_ANALYSIS.txt"
        # 'charmap' codec can't encode character '\U0001f3c6' in position 702: character maps to <undefined>
        with open(hybrid_path, 'w', encoding='utf-8') as f:
            f.write(hybrid_report)
        print(f"\n✓ Hybrid analysis report: {hybrid_path}")


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
        #'GTCRN_old': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\EXP3p1a\GTCRN_NOIZEUS_EARS_[{snr}]dB.csv",
        'GTCRN': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\GTCRN\GTCRN_EXP3p2a\GTCRN_EXP3p2a_merged_[{snr}]dB.csv",
        #'GTCRN_that_hybrid_uses': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\test_gtcrn_alone\GTCRN_SS_TEST2_[{snr}]dB.csv",
        #'GTCRN_SS_old': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\EXP3p1b_GTCRN_SS_pp_8ms_V0_f08\GTCRN_SS_TEST2_[{snr}]dB.csv",
        # 'GTCRN_SS_vad1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\EXP3p1b\EXP3p1b_GTCRN_SS_N4_lin_8ms_ov75_av1_nf1_f08_v1\GTCRN_SS_TEST2_[{snr}]dB.csv",
        # 'GTCRN_SS_delta': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\EXP3p1b\EXP3p1b_GTCRN_SS_delta15_N4_lin_8ms_ov75_av1_nf1_f08_v1\GTCRN_SS_TEST2_[{snr}]dB.csv",
        # 'WF_GTCRN_fr25_mu0.98_a_dd0.98_eta0.15': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\EXP3p1b_GTCRN_WF_ss\GTCRN_MWF_merged_[{snr}]dB.csv",
        'GTCRN_WF': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\GTCRN\GTCRNWF_EXP3p2a_25ms_quality\GTCRNWF_EXP3p2a_25ms_quality_merged_[{snr}]dB.csv",


        # Optimized ss_standalone 
        # Log, 20ms, 50% ovlp, floor 0.001, noisefr 1, Nband = 8  
        # --------mband_full_stream_hanning.py---------------
        # AVRGING = 1 
        #'mband_py_log_optimized_N8_20ms_ov50_fl0p001_noisefr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\optimal_SS_standalone\mband_py_N8_log_AVR1\mband_py_N8_log_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_lin_optimized_N8_20ms_ov50_fl0p001_noisefr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\optimal_SS_standalone\mband_py_N8_lin_AVR1\mband_py_N8_lin_AVR1_[{snr}]dB_MERGED.csv",
        #'mband_py_lin_optimized_N16_25ms_ov75_fl0p001_noisefr1': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP1\spectral\optimal_SS_standalone\lin_N16_25ms_ov75\mband_py_N16_lin_AVR1\mband_py_N16_lin_AVR1_[{snr}]dB_MERGED.csv",

        # Optimized hybrid
        # Log, 20ms, 75% ovlp, floor 0.7, noisefr 1, Nband = 4  
        # --------mband_full_stream_hanning.py---------------
        # AVRGING = 1 
        #'mband_py_log_hybrid_20ms_ov75_fl0p7_nf1_N4': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\objective\GTCRN_SS_TEST2_[{snr}]dB.csv",
        #'mband_py_log_hybrid_20ms_ov75_fl0p8_nf1_N4': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\log_20ms_ov75_fl08_N4\GTCRN_SS_TEST2_[{snr}]dB.csv",
        'GTCRN-SS': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\log_20ms_ov75_fl08_N4\GTCRN_SS_TEST2_[{snr}]dB.csv",

        #'mband_py_log_hybrid_20ms_ov50_fl0p87_v0_N4': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\log_20ms_ov50_fl07_N4_av0\GTCRN_SS_TEST2_[{snr}]dB.csv",
        #'mband_py_mel_hybrid_20ms_ov75_fl0p8_nf1_N4': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\mel_20ms_ov75_fl08_N4\GTCRN_SS_TEST2_[{snr}]dB.csv",
        #'mband_py_lin_hybrid_20ms_ov50_fl0p8_nf1_N4': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\lin_20ms_ov50_fl08_N4\GTCRN_SS_TEST2_[{snr}]dB.csv",
        #'lin_hybrid_v0_20ms_ov50_fl0p7_nf1_N4_av0': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\lin_20ms_ov50_fl07_N4_v0_av0\GTCRN_SS_TEST2_[{snr}]dB.csv",
        #'mband_py_log_hybrid_V0_20ms_ov75_fl0p8_nf1_N4': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\log_25ms_ov75_fl08_N4_v0\GTCRN_SS_TEST2_[{snr}]dB.csv",
        #'mband_py_log_hybrid_25ms_ov75_fl0p8_nf1_N4': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\log_25ms_ov75_fl08_N4\GTCRN_SS_TEST2_[{snr}]dB.csv",
        #'mband_py_log_hybrid_20ms_ov75_fl0p8_nf1_N8': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\log_20ms_ov75_fl08_N8\GTCRN_SS_TEST2_[{snr}]dB.csv",
        #'mband_py_lin_hybrid_20ms_ov75_fl0p7_nf1_N4': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\objective\linear\GTCRN_SS_TEST2_[{snr}]dB.csv",
        #'mband_py_mel_hybrid_20ms_ov75_fl0p7_nf1_N4': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\objective\mel\GTCRN_SS_TEST2_[{snr}]dB.csv",

        #'mband_py_mel_hybrid_20ms_ov50_fl0p3_nf1_N6': r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP3\spectral\GTCRN_SS\Optimal_hybrid\subjective\log\GTCRN_SS_TEST2_[{snr}]dB.csv",
    }
    
    # Set output folder
    output_folder = r"C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\compare_csvs\EXP3\spectral\noise_check_no_numbers"
    
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

