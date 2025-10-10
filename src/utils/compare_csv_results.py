import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class AudioEnhancementComparator:
    """
    Compare two audio enhancement methods across multiple metrics.
    Designed for comparing baseline vs VAD-integrated spectral subtraction.
    """
    
    def __init__(self, baseline_csv, enhanced_csv):
        """
        Initialize comparator with two CSV files.
        
        Parameters:
        -----------
        baseline_csv : str
            Path to baseline method results CSV
        enhanced_csv : str
            Path to enhanced method (with VAD) results CSV
        """
        self.df_baseline = pd.read_csv(baseline_csv)
        self.df_enhanced = pd.read_csv(enhanced_csv)
        
        # Extract file names from paths
        self.csv1_name = Path(baseline_csv).stem
        self.csv2_name = Path(enhanced_csv).stem
        
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
        
        # Merge dataframes on clean_file for comparison
        self.df_merged = self._merge_dataframes()
        
    def _merge_dataframes(self):
        """Merge baseline and enhanced dataframes on clean_file."""
        df_merged = self.df_baseline.merge(
            self.df_enhanced, 
            on='clean_file', 
            suffixes=('_csv1', '_csv2')
        )
        return df_merged
    
    def calculate_differences(self):
        """Calculate absolute and percentage differences for all metrics."""
        for metric in self.metrics:
            csv1_col = f'{metric}_csv1'
            csv2_col = f'{metric}_csv2'
            
            if csv1_col in self.df_merged.columns and csv2_col in self.df_merged.columns:
                # Absolute difference
                self.df_merged[f'{metric}_diff'] = (
                    self.df_merged[csv2_col] - self.df_merged[csv1_col]
                )
                
                # Percentage change
                self.df_merged[f'{metric}_pct_change'] = (
                    (self.df_merged[csv2_col] - self.df_merged[csv1_col]) / 
                    self.df_merged[csv1_col] * 100
                )
    
    def create_comparison_table(self):
        """
        Create a comprehensive comparison table with all metrics.
        
        Returns:
        --------
        pd.DataFrame with columns: Metric, CSV1_Avg, CSV2_Avg, Difference, Pct_Change
        """
        results = []
        
        for metric in self.metrics:
            csv1_col = f'{metric}_csv1'
            csv2_col = f'{metric}_csv2'
            
            if csv1_col in self.df_merged.columns and csv2_col in self.df_merged.columns:
                csv1_avg = self.df_merged[csv1_col].mean()
                csv2_avg = self.df_merged[csv2_col].mean()
                difference = csv2_avg - csv1_avg
                pct_change = (difference / csv1_avg) * 100
                
                results.append({
                    'Metric': metric,
                    f'{self.csv1_name}_Avg': round(csv1_avg, 4),
                    f'{self.csv2_name}_Avg': round(csv2_avg, 4),
                    'Difference': round(difference, 4),
                    'Pct_Change': f"{pct_change:+.2f}%"
                })
        
        return pd.DataFrame(results)
    
    def plot_metric_comparison_bars(self, figsize=(18, 10)):
        """
        Create bar chart comparison showing CSV1 vs CSV2 averages for all metrics.
        Separate subplots for different metric ranges.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        # Calculate differences if not already done
        if f'{self.metrics[0]}_diff' not in self.df_merged.columns:
            self.calculate_differences()
        
        # Group metrics by their ranges for better visualization
        group1 = ['PESQ', 'SI_SDR']  # Different ranges
        group2 = ['STOI']  # 0-1 range
        group3 = ['DNSMOS_p808_mos', 'DNSMOS_mos_sig', 'DNSMOS_mos_bak', 'DNSMOS_mos_ovr']  # 1-5 range
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.6, 1.2], hspace=0.4)
        
        # Plot Group 1: PESQ and SI_SDR
        ax1 = fig.add_subplot(gs[0])
        self._plot_metric_group(ax1, group1, 'PESQ & SI-SDR Comparison')
        
        # Plot Group 2: STOI
        ax2 = fig.add_subplot(gs[1])
        self._plot_metric_group(ax2, group2, 'STOI Comparison')
        
        # Plot Group 3: DNSMOS metrics
        ax3 = fig.add_subplot(gs[2])
        self._plot_metric_group(ax3, group3, 'DNSMOS Metrics Comparison')
        
        plt.suptitle(f'Metric Comparison: {self.csv1_name} vs {self.csv2_name}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def _plot_metric_group(self, ax, metrics, title):
        """Helper function to plot a group of metrics."""
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        csv1_avgs = []
        csv2_avgs = []
        
        for metric in metrics:
            csv1_col = f'{metric}_csv1'
            csv2_col = f'{metric}_csv2'
            
            if csv1_col in self.df_merged.columns and csv2_col in self.df_merged.columns:
                csv1_avgs.append(self.df_merged[csv1_col].mean())
                csv2_avgs.append(self.df_merged[csv2_col].mean())
            else:
                csv1_avgs.append(0)
                csv2_avgs.append(0)
        
        # Create bars
        bars1 = ax.bar(x_pos - width/2, csv1_avgs, width, 
                       label=self.csv1_name, color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, csv2_avgs, width, 
                       label=self.csv2_name, color='#e74c3c', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height != 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add metric ranges as horizontal lines
        for i, metric in enumerate(metrics):
            if metric in self.metric_ranges:
                min_val, max_val = self.metric_ranges[metric]
                # Add subtle range indicators
                ax.text(i, ax.get_ylim()[1] * 0.95, 
                       f'Range: [{min_val}, {max_val}]',
                       ha='center', fontsize=7, style='italic', color='gray')
        
        ax.set_xlabel('Metrics', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Value', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=0, ha='center')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    def generate_summary_report(self):
        """Generate a comprehensive text summary report."""
        # Calculate differences if not already done
        if f'{self.metrics[0]}_diff' not in self.df_merged.columns:
            self.calculate_differences()
        
        comparison_table = self.create_comparison_table()
        
        print("=" * 100)
        print("AUDIO ENHANCEMENT COMPARISON REPORT")
        print(f"CSV1: {self.csv1_name}")
        print(f"CSV2: {self.csv2_name}")
        print("=" * 100)
        print()
        
        print("METRIC COMPARISON TABLE:")
        print("-" * 100)
        print(comparison_table.to_string(index=False))
        print()
        
        print("\nDETAILED ANALYSIS:")
        print("-" * 100)
        
        for _, row in comparison_table.iterrows():
            metric = row['Metric']
            diff_col = f'{metric}_diff'
            
            if diff_col in self.df_merged.columns:
                improved = (self.df_merged[diff_col] > 0).sum()
                degraded = (self.df_merged[diff_col] < 0).sum()
                unchanged = (self.df_merged[diff_col] == 0).sum()
                total = improved + degraded + unchanged
                
                metric_range = self.metric_ranges[metric]
                
                print(f"\n{metric} (Range: {metric_range[0]} to {metric_range[1]}):")
                print(f"  Improved:  {improved}/{total} files ({improved/total*100:.1f}%)")
                print(f"  Degraded:  {degraded}/{total} files ({degraded/total*100:.1f}%)")
                print(f"  Unchanged: {unchanged}/{total} files ({unchanged/total*100:.1f}%)")
        
        print("\n" + "=" * 100)
    
    def export_detailed_comparison(self, output_path='comparison_results.csv'):
        """Export detailed comparison results to CSV."""
        self.df_merged.to_csv(output_path, index=False)
        print(f"Detailed comparison exported to: {output_path}")


# Example usage
if __name__ == "__main__":
    baseline_csv = "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\OldCSVFiles\\SS_EXP1p1b8ms50OVLP_han_remove_framefxn\\SS_EXP1p1b_merged_5dB.csv"
    enhanced_csv = "C:\\Users\\gabi\\Documents\\University\\Uni2025\\Investigation\\PROJECT-25P85\\Random\\OldCSVFiles\\SS_EXP2p1aGruModel05VAD\\SS_EXP1p2_merged_[5]dB.csv"

    comparator = AudioEnhancementComparator(
        baseline_csv=baseline_csv,
        enhanced_csv=enhanced_csv
    )
 
    # Calculate differences
    comparator.calculate_differences()
    
    # Generate report with comparison table
    comparator.generate_summary_report()
    
    # Create comparison table separately if needed
    print("\n\nCOMPARISON TABLE (for export):")
    comparison_table = comparator.create_comparison_table()
    print(comparison_table.to_string(index=False))
    
    # Save comparison table
    comparison_table.to_csv('metric_comparison_table.csv', index=False)
    print(f"\nComparison table saved to: metric_comparison_table.csv")
    
    # Create bar chart visualization
    fig = comparator.plot_metric_comparison_bars()
    plt.savefig('metric_comparison_bars.png', dpi=300, bbox_inches='tight')
    print("Bar chart saved to: metric_comparison_bars.png")
    
    # Export detailed results
    comparator.export_detailed_comparison()
    
    plt.show()