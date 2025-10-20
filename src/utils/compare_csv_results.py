import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime

class MultiAudioEnhancementComparator:
    """
    Compare up to 6 audio enhancement methods across multiple metrics.
    Flexible comparison tool for any audio enhancement CSV files.
    """
    
    def __init__(self, csv_files_dict, output_folder=None, experiment_name=None):
        """
        Initialize comparator with multiple CSV files.
        
        Parameters:
        -----------
        csv_files_dict : dict
            Dictionary with method names as keys and CSV file paths as values
        output_folder : str, optional
            Base output folder. If None, uses relative path to results/compare_csv
        experiment_name : str, optional
            Name for this comparison experiment (e.g., "wiener_vs_spectral")
        """
        if len(csv_files_dict) < 2:
            raise ValueError("Must provide at least 2 CSV files for comparison")
        if len(csv_files_dict) > 6:
            raise ValueError("Maximum 6 CSV files can be compared")
        
        self.method_names = list(csv_files_dict.keys())
        self.dataframes = {}
        
        # Setup output folder
        if output_folder is None:
            # Use relative path from script location
            script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
            self.output_folder = script_dir / "results" / "compare_csv"
        else:
            self.output_folder = Path(output_folder)
        
        # Create experiment-specific subfolder
        if experiment_name:
            self.experiment_name = experiment_name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = self.output_folder / f"{experiment_name}_{timestamp}"
        else:
            self.experiment_name = "comparison"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = self.output_folder / f"comparison_{timestamp}"
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {self.output_folder}")
        
        # Load all CSV files
        for method_name, csv_path in csv_files_dict.items():
            self.dataframes[method_name] = pd.read_csv(csv_path)
            print(f"Loaded {method_name}: {len(self.dataframes[method_name])} rows")
        
        # Metrics to analyze with their expected ranges
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
        
        # Color palette for different methods
        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        # Merge all dataframes
        self.df_merged = self._merge_dataframes()
        
        # Extract noise types for analysis
        self._extract_noise_types()
        
    def _merge_dataframes(self):
        """Merge all dataframes on clean_file for comparison."""
        # Start with first dataframe
        df_merged = self.dataframes[self.method_names[0]].copy()
        df_merged = df_merged.rename(columns={
            metric: f'{metric}_{self.method_names[0]}' 
            for metric in self.metrics if metric in df_merged.columns
        })
        
        # Merge remaining dataframes
        for method_name in self.method_names[1:]:
            df_temp = self.dataframes[method_name].copy()
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
        
        print(f"Merged dataset: {len(df_merged)} common files")
        return df_merged
    
    def _extract_noise_types(self):
        """Extract noise types from enhanced_file column for categorization."""
        # Try to extract noise type from filename
        if 'enhanced_file_x' in self.df_merged.columns:
            enhanced_col = 'enhanced_file_x'
        elif 'enhanced_file' in self.df_merged.columns:
            enhanced_col = 'enhanced_file'
        else:
            self.df_merged['noise_category'] = 'Unknown'
            return
        
        def categorize_noise(filename):
            filename_lower = str(filename).lower()
            
            # Construction noises (hardest)
            if any(x in filename_lower for x in ['jackhammer', 'drilling', 'crane', 'truck', 'construction']):
                return 'Construction'
            
            # Babble/Speech (hard)
            elif any(x in filename_lower for x in ['babble', 'cafeteria']):
                return 'Babble'
            
            # Train (moderate-hard)
            elif 'train' in filename_lower:
                return 'Train'
            
            # Street (moderate)
            elif 'street' in filename_lower:
                return 'Street'
            
            # Car (easier)
            elif 'car' in filename_lower or 'mph' in filename_lower:
                return 'Car'
            
            # Other stationary (easier)
            elif any(x in filename_lower for x in ['fan', 'cooler', 'flight', 'ssn']):
                return 'Stationary'
            
            else:
                return 'Other'
        
        self.df_merged['noise_category'] = self.df_merged[enhanced_col].apply(categorize_noise)
        
        # Print noise distribution
        print("\nNoise Category Distribution:")
        print(self.df_merged['noise_category'].value_counts())
    
    def calculate_differences(self, baseline_method=None):
        """
        Calculate differences relative to a baseline method.
        
        Parameters:
        -----------
        baseline_method : str, optional
            Name of baseline method. If None, uses first method.
        """
        if baseline_method is None:
            baseline_method = self.method_names[0]
        
        if baseline_method not in self.method_names:
            raise ValueError(f"Baseline method '{baseline_method}' not found")
        
        self.baseline_method = baseline_method
        
        for metric in self.metrics:
            baseline_col = f'{metric}_{baseline_method}'
            
            if baseline_col not in self.df_merged.columns:
                continue
            
            for method_name in self.method_names:
                if method_name == baseline_method:
                    continue
                
                method_col = f'{metric}_{method_name}'
                
                if method_col in self.df_merged.columns:
                    # Absolute difference
                    self.df_merged[f'{metric}_diff_{method_name}'] = (
                        self.df_merged[method_col] - self.df_merged[baseline_col]
                    )
                    
                    # Percentage change
                    self.df_merged[f'{metric}_pct_{method_name}'] = (
                        (self.df_merged[method_col] - self.df_merged[baseline_col]) / 
                        self.df_merged[baseline_col] * 100
                    )
    
    def create_comparison_table(self):
        """
        Create a comprehensive comparison table with all metrics and methods.
        
        Returns:
        --------
        pd.DataFrame with average values for each metric and method
        """
        results = []
        
        for metric in self.metrics:
            row = {'Metric': metric}
            
            for method_name in self.method_names:
                method_col = f'{metric}_{method_name}'
                
                if method_col in self.df_merged.columns:
                    avg_value = self.df_merged[method_col].mean()
                    row[method_name] = round(avg_value, 4)
                else:
                    row[method_name] = None
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def create_improvement_table(self, baseline_method=None):
        """
        Create table showing improvements over baseline with better formatting.
        Format: Method_Value | Change | %_Change for each method
        
        Parameters:
        -----------
        baseline_method : str, optional
            Name of baseline method. If None, uses first method.
        """
        if baseline_method is None:
            baseline_method = self.method_names[0]
        
        comparison_table = self.create_comparison_table()
        results = []
        
        for _, row in comparison_table.iterrows():
            metric = row['Metric']
            result_row = {'Metric': metric}
            
            # Add baseline value
            result_row[f'{baseline_method}'] = f"{row[baseline_method]:.4f}"
            
            # Add each method with value, absolute change, and percentage
            for method_name in self.method_names:
                if method_name == baseline_method:
                    continue
                
                if row[method_name] is not None and row[baseline_method] is not None:
                    value = row[method_name]
                    diff = value - row[baseline_method]
                    pct = (diff / row[baseline_method]) * 100
                    
                    # Create three separate columns for each method
                    result_row[f'{method_name}_Value'] = f"{value:.4f}"
                    result_row[f'{method_name}_Change'] = f"{diff:+.4f}"
                    result_row[f'{method_name}_%'] = f"{pct:+.2f}%"
                else:
                    result_row[f'{method_name}_Value'] = "N/A"
                    result_row[f'{method_name}_Change'] = "N/A"
                    result_row[f'{method_name}_%'] = "N/A"
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def create_noise_category_analysis(self, baseline_method=None):
        """
        Analyze performance by noise category (Construction, Babble, Train, etc.)
        
        Parameters:
        -----------
        baseline_method : str, optional
            Name of baseline method for comparison
        """
        if baseline_method is None:
            baseline_method = self.method_names[0]
        
        if 'noise_category' not in self.df_merged.columns:
            return None
        
        results = []
        
        for category in sorted(self.df_merged['noise_category'].unique()):
            category_data = self.df_merged[self.df_merged['noise_category'] == category]
            
            for metric in ['PESQ', 'SI_SDR', 'STOI', 'DNSMOS_mos_ovr']:
                row = {'Noise_Category': category, 'Metric': metric}
                
                for method_name in self.method_names:
                    method_col = f'{metric}_{method_name}'
                    if method_col in category_data.columns:
                        avg_value = category_data[method_col].mean()
                        
                        # Calculate improvement over baseline
                        if method_name != baseline_method:
                            baseline_col = f'{metric}_{baseline_method}'
                            baseline_avg = category_data[baseline_col].mean()
                            improvement = avg_value - baseline_avg
                            row[method_name] = f"{avg_value:.3f} ({improvement:+.3f})"
                        else:
                            row[method_name] = f"{avg_value:.3f}"
                
                results.append(row)
        
        return pd.DataFrame(results)
    
    def plot_metric_comparison_bars(self, figsize=(20, 12)):
        """
        Create bar chart comparison showing all methods for all metrics.
        """
        group1 = ['PESQ', 'SI_SDR']
        group2 = ['STOI']
        group3 = ['DNSMOS_p808_mos', 'DNSMOS_mos_sig', 'DNSMOS_mos_bak', 'DNSMOS_mos_ovr']
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.6, 1.2], hspace=0.4)
        
        ax1 = fig.add_subplot(gs[0])
        self._plot_metric_group(ax1, group1, 'PESQ & SI-SDR Comparison')
        
        ax2 = fig.add_subplot(gs[1])
        self._plot_metric_group(ax2, group2, 'STOI Comparison')
        
        ax3 = fig.add_subplot(gs[2])
        self._plot_metric_group(ax3, group3, 'DNSMOS Metrics Comparison')
        
        plt.suptitle(f'{self.experiment_name}: Multi-Method Comparison', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def _plot_metric_group(self, ax, metrics, title):
        """Helper function to plot a group of metrics."""
        num_methods = len(self.method_names)
        x_pos = np.arange(len(metrics))
        width = 0.8 / num_methods
        
        all_data = []
        for method_name in self.method_names:
            method_avgs = []
            for metric in metrics:
                method_col = f'{metric}_{method_name}'
                if method_col in self.df_merged.columns:
                    method_avgs.append(self.df_merged[method_col].mean())
                else:
                    method_avgs.append(0)
            all_data.append(method_avgs)
        
        for i, (method_name, method_data) in enumerate(zip(self.method_names, all_data)):
            offset = (i - num_methods/2 + 0.5) * width
            bars = ax.bar(x_pos + offset, method_data, width, 
                         label=method_name, 
                         color=self.colors[i], 
                         alpha=0.8, 
                         edgecolor='black')
            
            for bar in bars:
                height = bar.get_height()
                if height != 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', 
                           fontsize=7 if num_methods > 3 else 9, 
                           fontweight='bold')
        
        for i, metric in enumerate(metrics):
            if metric in self.metric_ranges:
                min_val, max_val = self.metric_ranges[metric]
                ax.text(i, ax.get_ylim()[1] * 0.95, 
                       f'Range: [{min_val}, {max_val}]',
                       ha='center', fontsize=7, style='italic', color='gray')
        
        ax.set_xlabel('Metrics', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Value', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=0, ha='center')
        ax.legend(loc='upper left', fontsize=9, ncol=2 if num_methods > 3 else 1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    def plot_noise_category_comparison(self, metric='PESQ', figsize=(14, 8)):
        """
        Plot performance by noise category to show which noises are hardest.
        
        Parameters:
        -----------
        metric : str
            Metric to plot (default: 'PESQ')
        figsize : tuple
            Figure size
        """
        if 'noise_category' not in self.df_merged.columns:
            print("Noise category information not available")
            return None
        
        categories = sorted(self.df_merged['noise_category'].unique())
        num_methods = len(self.method_names)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x_pos = np.arange(len(categories))
        width = 0.8 / num_methods
        
        for i, method_name in enumerate(self.method_names):
            method_col = f'{metric}_{method_name}'
            if method_col not in self.df_merged.columns:
                continue
            
            category_means = []
            for category in categories:
                category_data = self.df_merged[self.df_merged['noise_category'] == category]
                category_means.append(category_data[method_col].mean())
            
            offset = (i - num_methods/2 + 0.5) * width
            bars = ax.bar(x_pos + offset, category_means, width,
                         label=method_name,
                         color=self.colors[i],
                         alpha=0.8,
                         edgecolor='black')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom',
                       fontsize=8,
                       fontweight='bold')
        
        ax.set_xlabel('Noise Category', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Performance by Noise Category\n(Comparing Difficulty Levels)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self, baseline_method=None):
        """Generate a comprehensive text summary report."""
        if baseline_method is None:
            baseline_method = self.method_names[0]
        
        comparison_table = self.create_comparison_table()
        improvement_table = self.create_improvement_table(baseline_method)
        
        report_lines = []
        report_lines.append("=" * 120)
        report_lines.append(f"EXPERIMENT: {self.experiment_name.upper()}")
        report_lines.append("MULTI-METHOD AUDIO ENHANCEMENT COMPARISON REPORT")
        report_lines.append("=" * 120)
        report_lines.append(f"Methods compared: {', '.join(self.method_names)}")
        report_lines.append(f"Baseline method: {baseline_method}")
        report_lines.append(f"Number of common files: {len(self.df_merged)}")
        report_lines.append(f"Output folder: {self.output_folder}")
        report_lines.append("=" * 120)
        report_lines.append("")
        
        report_lines.append("AVERAGE METRIC VALUES:")
        report_lines.append("-" * 120)
        report_lines.append(comparison_table.to_string(index=False))
        report_lines.append("")
        
        report_lines.append("\nIMPROVEMENT OVER BASELINE:")
        report_lines.append("Format: Value |(Absolute Change) | % (Percentage Change)")
        report_lines.append("-" * 120)
        report_lines.append(improvement_table.to_string(index=False))
        report_lines.append("")
        
        report_lines.append("\nFILE-LEVEL IMPROVEMENT STATISTICS:")
        report_lines.append("-" * 120)
        
        for metric in self.metrics:
            baseline_col = f'{metric}_{baseline_method}'
            if baseline_col not in self.df_merged.columns:
                continue
            
            report_lines.append(f"\n{metric} (Range: {self.metric_ranges[metric][0]} to {self.metric_ranges[metric][1]}):")
            
            for method_name in self.method_names:
                if method_name == baseline_method:
                    continue
                
                diff_col = f'{metric}_diff_{method_name}'
                if diff_col in self.df_merged.columns:
                    improved = (self.df_merged[diff_col] > 0).sum()
                    degraded = (self.df_merged[diff_col] < 0).sum()
                    unchanged = (self.df_merged[diff_col] == 0).sum()
                    total = improved + degraded + unchanged
                    
                    report_lines.append(f"  {method_name}:")
                    report_lines.append(f"    Improved:  {improved}/{total} files ({improved/total*100:.1f}%)")
                    report_lines.append(f"    Degraded:  {degraded}/{total} files ({degraded/total*100:.1f}%)")
                    report_lines.append(f"    Unchanged: {unchanged}/{total} files ({unchanged/total*100:.1f}%)")
        
        # Add noise category analysis if available
        if 'noise_category' in self.df_merged.columns:
            report_lines.append("\n\nPERFORMANCE BY NOISE CATEGORY (Hardest Cases):")
            report_lines.append("-" * 120)
            noise_analysis = self.create_noise_category_analysis(baseline_method)
            if noise_analysis is not None:
                report_lines.append(noise_analysis.to_string(index=False))
        
        report_lines.append("\n" + "=" * 120)
        
        # Print to console
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # Save to file
        report_path = self.output_folder / f"{self.experiment_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {report_path}")
        
        return report_text
    
    def export_all_results(self):
        """Export all comparison results and visualizations."""
        print("\n" + "="*80)
        print("EXPORTING RESULTS...")
        print("="*80)
        
        # 1. Comparison table
        comparison_table = self.create_comparison_table()
        comparison_path = self.output_folder / f"{self.experiment_name}_comparison_table.csv"
        comparison_table.to_csv(comparison_path, index=False)
        print(f"✓ Comparison table: {comparison_path}")
        
        # 2. Improvement table
        improvement_table = self.create_improvement_table()
        improvement_path = self.output_folder / f"{self.experiment_name}_improvement_table.csv"
        improvement_table.to_csv(improvement_path, index=False)
        print(f"✓ Improvement table: {improvement_path}")
        
        # 3. Noise category analysis
        if 'noise_category' in self.df_merged.columns:
            noise_analysis = self.create_noise_category_analysis()
            if noise_analysis is not None:
                noise_path = self.output_folder / f"{self.experiment_name}_noise_category_analysis.csv"
                noise_analysis.to_csv(noise_path, index=False)
                print(f"✓ Noise category analysis: {noise_path}")
        
        # 4. Detailed merged data
        detailed_path = self.output_folder / f"{self.experiment_name}_detailed_comparison.csv"
        self.df_merged.to_csv(detailed_path, index=False)
        print(f"✓ Detailed comparison: {detailed_path}")
        
        # 5. Bar chart
        fig1 = self.plot_metric_comparison_bars()
        bar_path = self.output_folder / f"{self.experiment_name}_bar_comparison.png"
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"✓ Bar chart: {bar_path}")
    
        
        # 7. Noise category comparison plots for key metrics
        for metric in ['PESQ', 'SI_SDR', 'STOI']:
            fig = self.plot_noise_category_comparison(metric=metric)
            if fig is not None:
                noise_plot_path = self.output_folder / f"{self.experiment_name}_noise_{metric}.png"
                plt.savefig(noise_plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Noise comparison ({metric}): {noise_plot_path}")
        
        print("="*80)
        print(f"All results saved to: {self.output_folder}")
        print("="*80)


# Example usage
if __name__ == "__main__":
    # Define your CSV files with descriptive names
    csv_files = {
        'Baseline_Noisy': "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP0\\noisy_vs_clean\\BASELINE_merged_SNR[10]dB.csv",
        # 'Wiener_Filter':  "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP1\\wiener\\WF_EXP1p1b\\WF_EXP1p1b_merged_[5]dB.csv",
        # 'Spectral_Subtraction': "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP1\\spectral\\SS_EXP1p1b\\SS_EXP1p1b_merged_5dB.csv",
        'GTCRN_baseline': "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP3\\EXP3p1a\\GTCRN_NOIZEUS_EARS_[5]dB.csv",
        # 'GTCRN_SS': "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP3\\EXP3p1b\\GTCRN_SS_NOIZEUS_EARS_[5]dB.csv",
        'SS_GTCRN': "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP3\\EXP3p1c\\SS_GTCRN_[5]dB.csv",
        'GTCRN_SS_wovad_lowerfloor': "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP3\\EXP3p1b_wovad9\\GTCRN_SS_TEST2_[5]dB.csv",
        'GTCRN_WF': "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP3\\GTCRN\\GTCRN_MWF\\GTCRN_MWF_merged_[5]dB.csv",
        'GTCRN_WF_eta08_25ms': "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\EXP3\\GTCRN\\GTCRN_MWF_eta08\\GTCRN_MWF_merged_[5]dB.csv",
        
        # Add up to 6 total methods
    }
    
    # Set output folder (will create subdirectories)
    output_folder = "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\results\\compare_csvs\\compare_GTCRNS_hybrids_wwovad8_20ms"
    
    # Create comparator with experiment name
    comparator = MultiAudioEnhancementComparator(
        csv_files,
        output_folder=output_folder,
        experiment_name="GTCRNs_5dB"
    )
    
    # Calculate differences (relative to baseline - first method by default)
    comparator.calculate_differences(baseline_method='Baseline_Noisy')
    
    # Generate comprehensive report
    comparator.generate_summary_report(baseline_method='Baseline_Noisy')
    
    # Export all results and visualizations
    comparator.export_all_results()
    
    print("\n Analysis complete! Check the output folder for all results.")