#!/usr/bin/env python3
"""
Combined bifidoAnnotator: Complete pipeline for the annotation of bifidobacterial enzymes involved in HMG-utilization
and publication-ready heatmap generation with adaptive sizing.

This tool performs hierarchical annotation of GH-encoding genes using MMseqs2,
generates data matrices, and creates sophisticated heatmaps with automatic size optimization.

Author: Nicholas Pucci & Daniel R. Mende (modified and combined)
Version: 1.0.1 - Updated with auto-download, Python 3.10+ requirement, and ColorBrewer Dark2 palettes
"""

import sys

# Check Python version requirement
if sys.version_info < (3, 10):
    sys.stderr.write("Error: bifidoAnnotator requires Python 3.10 or higher.\n")
    sys.stderr.write(f"You are using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")
    sys.stderr.write("Please upgrade your Python version or use a compatible environment.\n")
    sys.exit(1)

import argparse
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
import datetime
import time
import logging
import urllib.request
import tarfile
import hashlib

warnings.filterwarnings('ignore')

# Suppress matplotlib font warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Global variables for logging
LOG_FILE = None
START_TIME = None

# Configuration parameters
COVERAGE_THRESHOLD = 0.5
BITSCORE_THRESHOLD = 200

# Database download configuration
DEFAULT_DB_DIR = os.path.expanduser('~/.bifidoannotator/database')
ZENODO_URL = "https://zenodo.org/records/17206993/files/bifDB_dir.tar.gz"
ZENODO_MD5 = "ed4606903275bfc01ac14913d5711529"

def download_database():
    """Download and extract database from Zenodo on first use"""
    db_path = os.path.join(DEFAULT_DB_DIR, 'bifDB_dir', 'bifDB')
    mapping_path = os.path.join(DEFAULT_DB_DIR, 'mapping_file.tsv')
    marker_file = os.path.join(DEFAULT_DB_DIR, '.downloaded')
    
    # Check if already downloaded
    if os.path.exists(marker_file) and os.path.exists(db_path):
        return True
    
    print("=" * 80)
    print("FIRST RUN: Downloading reference database from Zenodo")
    print("This is a one-time download (~139 MB compressed, ~350 MB extracted)")
    print(f"Downloading from: {ZENODO_URL}")
    print("=" * 80)
    
    try:
        # Create directory
        os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
        
        # Download with progress
        tar_file = os.path.join(DEFAULT_DB_DIR, 'bifDB_dir.tar.gz')
        
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        
        urllib.request.urlretrieve(ZENODO_URL, tar_file, reporthook=report_progress)
        print()  # New line after progress
        
        # Verify checksum
        print("Verifying download integrity...")
        md5_hash = hashlib.md5()
        with open(tar_file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        if md5_hash.hexdigest() != ZENODO_MD5:
            print(f"ERROR: Checksum mismatch! Download may be corrupted.")
            os.remove(tar_file)
            return False
        
        print("Checksum verified!")
        
        # Extract
        print("Extracting database...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(path=DEFAULT_DB_DIR)
        
        # Clean up tar file
        os.remove(tar_file)
        
        # Create marker file
        with open(marker_file, 'w') as f:
            f.write('downloaded\n')
        
        print("Database setup complete!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\nERROR: Failed to download database: {e}")
        print("Please check your internet connection or download manually from:")
        print(ZENODO_URL)
        return False

def get_default_paths():
    """Get default paths for database and mapping file"""
    # First check if database exists in default location
    db_dir = DEFAULT_DB_DIR
    bifdb_path = os.path.join(db_dir, 'bifDB_dir', 'bifDB')
    mapping_path = os.path.join(db_dir, 'mapping_file.tsv')
    
    # Try to get mapping file from package installation (if available)
    try:
        import pkg_resources
        package_mapping = pkg_resources.resource_filename('bifidoannotator', 'database/mapping_file.tsv')
        if os.path.exists(package_mapping):
            mapping_path = package_mapping
    except:
        pass
    
    return bifdb_path, mapping_path

def initialize_log(output_dir, args):
    """Initialize the log file"""
    global LOG_FILE, START_TIME
    START_TIME = time.time()
    
    LOG_FILE = os.path.join(output_dir, 'bifidoAnnotator_log.txt')
    
    with open(LOG_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMBINED BIFIDOANNOTATOR: Complete GH Annotation & Visualization Pipeline\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command line: {' '.join(sys.argv)}\n")
        f.write("\n")
        
        # Log parameters
        f.write("PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Reference database: {args.bifdb}\n")
        f.write(f"Mapping file: {args.mapping_file}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Annotations file: {args.annotations_file if args.annotations_file else 'None (basic heatmaps)'}\n")
        f.write(f"Threads: {args.threads}\n")
        f.write(f"MMseqs2 sensitivity: {args.sensitivity}\n")
        f.write(f"Coverage threshold: {COVERAGE_THRESHOLD}\n")
        f.write(f"Bitscore threshold: {BITSCORE_THRESHOLD}\n")
        
        # Figure size parameters
        f.write(f"GH figure size: {'Auto-adaptive' if not args.gh_figsize else f'{args.gh_figsize[0]}x{args.gh_figsize[1]}'}\n")
        f.write(f"Cluster figure size: {'Auto-adaptive' if not args.cluster_figsize else f'{args.cluster_figsize[0]}x{args.cluster_figsize[1]}'}\n")
        f.write(f"Enzyme figure size: {'Auto-adaptive' if not args.enzyme_figsize else f'{args.enzyme_figsize[0]}x{args.enzyme_figsize[1]}'}\n")
        f.write(f"Heatmap color scheme: {args.heatmap_col}\n")
        f.write("\n")

def log_message(message, print_also=True):
    """Write message to log file and optionally print"""
    if print_also:
        print(message)
    
    if LOG_FILE:
        with open(LOG_FILE, 'a') as f:
            f.write(message + "\n")

def log_section(title):
    """Log a section header"""
    message = f"\n{title.upper()}\n" + "-" * len(title) + "\n"
    log_message(message, print_also=False)

def finalize_log(combined_results, genome_names, matrices):
    """Write final statistics to log file"""
    global START_TIME
    
    if not LOG_FILE:
        return
    
    end_time = time.time()
    runtime = end_time - START_TIME
    
    with open(LOG_FILE, 'a') as f:
        log_section("FINAL SUMMARY")
        
        f.write(f"Analysis completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)\n\n")
        
        if len(combined_results) > 0:
            # Overall statistics
            f.write("OVERALL ANNOTATION STATISTICS:\n")
            f.write(f"Total genomes processed: {len(genome_names)}\n")
            f.write(f"Genomes with annotations: {combined_results['Genome'].nunique()}\n")
            f.write(f"Genomes without annotations: {len(genome_names) - combined_results['Genome'].nunique()}\n")
            f.write(f"Total sequences annotated: {len(combined_results)}\n")
            f.write(f"Unique GH families detected: {combined_results['GH_family'].nunique()}\n")
            f.write(f"Unique clusters detected: {combined_results['Assigned_cluster'].nunique()}\n\n")
            
            # GH family breakdown
            f.write("GH FAMILY DISTRIBUTION:\n")
            gh_family_counts = combined_results['GH_family'].value_counts()
            for family, count in gh_family_counts.head(10).items():
                percentage = (count / len(combined_results)) * 100
                f.write(f"  {family}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Validation status
            f.write("VALIDATION STATUS BREAKDOWN:\n")
            validation_counts = combined_results['Validation_status'].value_counts()
            for status, count in validation_counts.items():
                percentage = (count / len(combined_results)) * 100
                f.write(f"  {status}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Quality metrics
            f.write("ANNOTATION QUALITY:\n")
            f.write(f"  Mean percent identity: {combined_results['pident'].mean():.1f}%\n")
            f.write(f"  Mean bit score: {combined_results['bits'].mean():.1f}\n")
            f.write(f"  Identity range: {combined_results['pident'].min():.1f}% - {combined_results['pident'].max():.1f}%\n\n")
            
        else:
            f.write("No annotations found across all genomes.\n\n")
        
        # Output files
        f.write("OUTPUT FILES GENERATED:\n")
        output_files = []
        tables_dir = os.path.join(os.path.dirname(LOG_FILE), 'bifidoAnnotator_tables')
        vis_dir = os.path.join(os.path.dirname(LOG_FILE), 'bifidoAnnotator_visualizations')
        
        for dir_path in [tables_dir, vis_dir]:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        size_kb = os.path.getsize(file_path) / 1024
                        f.write(f"  {file}: {size_kb:.1f} KB\n")
        
        f.write(f"\nLog file: bifidoAnnotator_log.txt\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Analysis completed successfully!\n")
        f.write("=" * 80 + "\n")

# Set matplotlib parameters for publication-quality figures
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 14

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        subprocess.run(['mmseqs', '-h'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("=" * 80)
        print("ERROR: MMseqs2 not found")
        print("=" * 80)
        print("\nMMseqs2 is required but not installed in your PATH.")
        print("\nTo install MMseqs2:")
        print("  • Ubuntu/Debian: sudo apt-get install mmseqs2")
        print("  • macOS: brew install mmseqs2")
        print("  • Conda: conda install -c bioconda mmseqs2")
        print("\nFor more information: https://github.com/soedinglab/MMseqs2")
        print("=" * 80)
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    # Get default paths
    default_bifdb, default_mapping = get_default_paths()
    
    parser = argparse.ArgumentParser(
        description="Combined bifidoAnnotator: Complete GH annotation and visualization pipeline with adaptive sizing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single genome (database auto-downloaded on first run)
  bifidoAnnotator -i genome.fasta -o results

  # Batch processing
  bifidoAnnotator -d genomes_dir -s sample_list.txt -o results
  
  # With annotations
  bifidoAnnotator -i genome.fasta --annotations_file metadata.tsv -o results
  
  # Using custom database
  bifidoAnnotator -i genome.fasta --bifdb /custom/db --mapping_file /custom/mapping.tsv -o results

Note: On first run, the reference database (~350 MB) will be automatically 
downloaded from Zenodo. This only happens once.
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input_file', 
                           help='Path to single input FASTA file')
    input_group.add_argument('-d', '--genome_directory',
                           help='Path to directory containing input FASTA files')
    
    parser.add_argument('-s', '--sample_file',
                       help='Text file listing genome names for processing (required with -d)')
    parser.add_argument('-o', '--output_dir', default='bifidoAnnotator_output',
                       help='Output directory (default: bifidoAnnotator_output)')
    
    # Database arguments - now OPTIONAL with defaults
    parser.add_argument('--bifdb',
                       default=default_bifdb,
                       help='Path to MMseqs2 database (default: auto-download from Zenodo on first run)')
    parser.add_argument('--mapping_file',
                       default=default_mapping,
                       help='Path to mapping file (default: packaged or downloaded with database)')
    
    # Optional arguments
    parser.add_argument('--annotations_file',
                       help='TSV file with genome annotations for heatmap legends')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads for MMseqs2 (default: 4)')
    parser.add_argument('--sensitivity', type=float, default=7.5,
                       help='MMseqs2 sensitivity (default: 7.5)')
    
    # Visualization parameters
    parser.add_argument('--gh-figsize', nargs=2, type=int, default=None,
                       help='GH heatmap figure size (width height)')
    parser.add_argument('--cluster-figsize', nargs=2, type=int, default=None,
                       help='Cluster heatmap figure size (width height)')  
    parser.add_argument('--enzyme-figsize', nargs=2, type=int, default=None,
                       help='Enzyme heatmap figure size (width height)')
    parser.add_argument('-hc', '--heatmap_col', type=str, default='blue', choices=['red', 'blue'],
                       help='Color scheme for heatmap and annotations (default: blue)')
    
    args = parser.parse_args()
    
    # Check if database needs to be downloaded
    if not os.path.exists(args.bifdb):
        print(f"Database not found at: {args.bifdb}")
        if not download_database():
            print("\nERROR: Could not download database automatically.")
            print("\nManual download instructions:")
            print(f"1. Download: {ZENODO_URL}")
            print(f"2. Extract to: {DEFAULT_DB_DIR}")
            print(f"3. Verify bifDB file exists at: {os.path.join(DEFAULT_DB_DIR, 'bifDB_dir', 'bifDB')}")
            sys.exit(1)
        
        # Update paths after download
        args.bifdb, args.mapping_file = get_default_paths()
    
    # Final validation
    if not os.path.exists(args.bifdb):
        print(f"ERROR: Database file not found: {args.bifdb}")
        sys.exit(1)
    
    if not os.path.exists(args.mapping_file):
        print(f"ERROR: Mapping file not found: {args.mapping_file}")
        print("\nThe mapping file should be included with the package or downloaded with the database.")
        sys.exit(1)
    
    return args

def run_mmseqs_search(query_file, ref_db, output_prefix, threads, sensitivity):
    """Run MMseqs2 search against reference database"""
    log_message(f"Running MMseqs2 search for {query_file}...")
    
    # Create temporary directories
    tmp_dir = f"{output_prefix}_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # MMseqs2 search command
    cmd = [
        'mmseqs', 'easy-search',
        query_file, ref_db,
        f"{output_prefix}_results.tsv",
        tmp_dir,
        '--format-output', 'query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits',
        '--threads', str(threads),
        '-s', str(sensitivity)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log_message(f"MMseqs2 search completed for {query_file}")
    except subprocess.CalledProcessError as e:
        error_msg = f"ERROR: MMseqs2 search failed for {query_file}\nCommand: {' '.join(cmd)}\nError: {e.stderr.decode()}"
        log_message(error_msg)
        return False
    
    # Clean up temporary directory
    subprocess.run(['rm', '-rf', tmp_dir], check=True)
    return True

def load_mapping_file(mapping_file):
    """Load and process the mapping file"""
    log_message(f"Loading mapping file: {mapping_file}")
    try:
        mapping_df = pd.read_csv(mapping_file, sep='\t')
        log_message(f"Loaded {len(mapping_df)} reference sequences")
        
        # Check for required threshold columns
        required_cols = ['GH_family-F1_threshold', 'GH-cluster threshold']
        missing_cols = [col for col in required_cols if col not in mapping_df.columns]
        if missing_cols:
            error_msg = f"ERROR: Missing required threshold columns: {missing_cols}\nExpected columns: GH_family-F1_threshold, GH-cluster threshold"
            log_message(error_msg)
            sys.exit(1)
        
        # Convert threshold columns to numeric
        for col in required_cols:
            mapping_df[col] = pd.to_numeric(mapping_df[col], errors='coerce')
            n_missing = mapping_df[col].isna().sum()
            if n_missing > 0:
                log_message(f"WARNING: {n_missing} rows have missing/invalid values in {col}")
        
        # Log statistics about mapping file
        log_section("MAPPING FILE STATISTICS")
        log_message(f"Total reference sequences: {len(mapping_df)}", print_also=False)
        log_message(f"Mapping file columns: {list(mapping_df.columns)}", print_also=False)
        gh_families = mapping_df['GH_family'].value_counts()
        log_message(f"GH families in database: {len(gh_families)}", print_also=False)
        for family, count in gh_families.head(5).items():
            log_message(f"  {family}: {count} sequences", print_also=False)
        
        validation_counts = mapping_df['Validation_status'].value_counts()
        log_message("Validation status distribution:", print_also=False)
        for status, count in validation_counts.items():
            percentage = (count / len(mapping_df)) * 100
            log_message(f"  {status}: {count} ({percentage:.1f}%)", print_also=False)
        
        log_message("Mapping file loaded successfully")
        return mapping_df
        
    except Exception as e:
        log_message(f"ERROR: Failed to load mapping file: {e}")
        sys.exit(1)

def process_mmseqs_results(results_file, mapping_df, genome_name):
    """Process MMseqs2 results and apply annotation thresholds"""
    if not os.path.exists(results_file):
        log_message(f"WARNING: Results file not found: {results_file}")
        return pd.DataFrame()
    
    # Load MMseqs2 results
    try:
        results_df = pd.read_csv(results_file, sep='\t', 
                                names=['query', 'target', 'pident', 'alnlen', 'mismatch', 
                                      'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits'])
    except Exception as e:
        log_message(f"WARNING: Failed to load results file {results_file}: {e}")
        return pd.DataFrame()
    
    if len(results_df) == 0:
        log_message(f"WARNING: No hits found for {genome_name}")
        return pd.DataFrame()
    
    # Log processing details
    log_message(f"  Raw hits found: {len(results_df)}", print_also=False)
    
    # Keep only best hit per query
    results_df = results_df.loc[results_df.groupby('query')['bits'].idxmax()]
    log_message(f"  Unique query sequences: {len(results_df)}", print_also=False)
    
    # Merge with mapping information
    results_df = results_df.merge(mapping_df, left_on='target', right_on='Protein_Name', how='left')
    
    # Calculate coverage
    results_df['query_coverage'] = (results_df['qend'] - results_df['qstart'] + 1) / results_df['alnlen']
    results_df['target_coverage'] = (results_df['tend'] - results_df['tstart'] + 1) / results_df['alnlen']
    results_df['min_coverage'] = np.minimum(results_df['query_coverage'], results_df['target_coverage'])
    
    # Set thresholds from configuration
    min_coverage = COVERAGE_THRESHOLD
    min_bitscore = BITSCORE_THRESHOLD
    
    # Apply reference-specific thresholds for annotation
    annotations = []
    failed_coverage = 0
    failed_bitscore = 0
    failed_family_threshold = 0
    missing_threshold_data = 0
    
    for _, row in results_df.iterrows():
        # Check if we have threshold information
        if pd.isna(row.get('GH_family-F1_threshold')) or pd.isna(row.get('GH-cluster threshold')):
            missing_threshold_data += 1
            continue
        
        # Get reference-specific thresholds
        gh_family_threshold = float(row['GH_family-F1_threshold'])
        cluster_threshold = float(row['GH-cluster threshold'])
        
        # Check coverage
        if row['min_coverage'] < min_coverage:
            failed_coverage += 1
            continue
        
        # Check bitscore
        if row['bits'] < min_bitscore:
            failed_bitscore += 1
            continue
        
        # Check GH family assignment
        if row['pident'] >= gh_family_threshold:
            annotation = row.copy()
            annotation['Genome'] = genome_name
            
            # Check for cluster assignment
            if row['pident'] >= cluster_threshold:
                annotation['Assigned_cluster'] = row['Cluster_annotation']
            else:
                gh_family = row['GH_family']
                annotation['Assigned_cluster'] = f"{gh_family}_cluster_undefined"
            
            annotations.append(annotation)
        else:
            failed_family_threshold += 1
    
    if annotations:
        filtered_df = pd.DataFrame(annotations)
        log_message(f"Processed {len(filtered_df)} annotations for {genome_name}")
        
        # Print summary
        n_family_only = len(filtered_df[filtered_df['Assigned_cluster'].str.contains('_cluster_undefined')])
        n_cluster = len(filtered_df[~filtered_df['Assigned_cluster'].str.contains('_cluster_undefined')])
        log_message(f"  - {n_family_only} sequences: GH family assignment only")
        log_message(f"  - {n_cluster} sequences: GH family + cluster assignment")
        
        # Log filtering statistics
        log_message(f"  - Failed coverage filter: {failed_coverage}", print_also=False)
        log_message(f"  - Failed bitscore filter: {failed_bitscore}", print_also=False)
        log_message(f"  - Failed family threshold: {failed_family_threshold}", print_also=False)
        log_message(f"  - Missing threshold data: {missing_threshold_data}", print_also=False)
        
        # Log quality statistics
        log_message(f"  - Mean percent identity: {filtered_df['pident'].mean():.1f}%", print_also=False)
        log_message(f"  - Mean coverage: {filtered_df['min_coverage'].mean():.2f}", print_also=False)
        
        return filtered_df
    else:
        log_message(f"No annotations passed thresholds for {genome_name}")
        log_message(f"  - Failed coverage filter: {failed_coverage}", print_also=False)
        log_message(f"  - Failed bitscore filter: {failed_bitscore}", print_also=False)
        log_message(f"  - Failed family threshold: {failed_family_threshold}", print_also=False)
        log_message(f"  - Missing threshold data: {missing_threshold_data}", print_also=False)
        return pd.DataFrame()

def create_output_structure(output_dir):
    """Create output directory structure"""
    subdirs = ['bifidoAnnotator_tables', 'bifidoAnnotator_visualizations']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

def generate_detailed_annotations(all_results, output_dir):
    """Generate detailed annotation tables"""
    if len(all_results) == 0:
        log_message("WARNING: No results to process")
        return
    
    # Base columns
    base_columns = ['query', 'Genome', 'GH_family', 'Enzyme', 'Cluster_ID', 
                   'Assigned_cluster', 'Validation_status', 'Reference',
                   'HMG-utilization', 'pident', 'bits', 'evalue']
    
    # Select columns that exist
    available_columns = list(all_results.columns)
    final_columns = [col for col in base_columns if col in available_columns]
    detailed_df = all_results[final_columns].copy()
    
    detailed_df.to_csv(os.path.join(output_dir, 'bifidoAnnotator_tables', 'detailed_annotations.tsv'), 
                      sep='\t', index=False)
    log_message("Generated detailed annotations table")

def generate_genome_summary(all_results, output_dir, all_genome_names):
    """Generate per-genome summary with copy numbers"""
    if len(all_results) == 0:
        return
    
    # Get all available columns
    available_columns = list(all_results.columns)
    
    # Define columns to exclude from aggregation
    exclude_from_agg = [
        'query', 'target', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 
        'tstart', 'tend', 'pident', 'bits', 'evalue', 'query_coverage', 
        'target_coverage', 'min_coverage', 'GH_family-F1_threshold', 
        'GH-cluster threshold', 'Protein_Name'
    ]
    
    # Build aggregation dictionary
    agg_dict = {'query': 'count'}
    
    # Add all non-excluded columns
    additional_agg_columns = []
    for col in available_columns:
        if (col not in exclude_from_agg and 
            col not in ['Genome', 'GH_family', 'Enzyme', 'Assigned_cluster']):
            agg_dict[col] = 'first'
            additional_agg_columns.append(col)
    
    log_message(f"Aggregating {len(additional_agg_columns)} additional columns in genome summary:", print_also=False)
    if additional_agg_columns:
        log_message(f"  Additional columns: {sorted(additional_agg_columns)}", print_also=False)
    
    # Count copies per genome per cluster
    genome_summary = all_results.groupby(['Genome', 'GH_family', 'Enzyme', 'Assigned_cluster']).agg(agg_dict).reset_index()
    genome_summary.rename(columns={'query': 'copy_number'}, inplace=True)
    
    # Add rows for genomes with no annotations
    if all_genome_names:
        genomes_with_annotations = set(all_results['Genome'].unique())
        missing_genomes = set(all_genome_names) - genomes_with_annotations
        
        if missing_genomes:
            log_message(f"Adding {len(missing_genomes)} genomes with no annotations to genome summary (with NA values)")
            
            missing_rows = []
            for genome in missing_genomes:
                missing_row = {
                    'Genome': genome,
                    'GH_family': pd.NA,
                    'Enzyme': pd.NA,
                    'Assigned_cluster': pd.NA,
                    'copy_number': pd.NA
                }
                
                for col in additional_agg_columns:
                    missing_row[col] = pd.NA
                
                missing_rows.append(missing_row)
            
            missing_df = pd.DataFrame(missing_rows)
            genome_summary = pd.concat([genome_summary, missing_df], ignore_index=True)
    
    genome_summary.to_csv(os.path.join(output_dir, 'bifidoAnnotator_tables', 'genome_summary.tsv'), 
                         sep='\t', index=False)
    log_message("Generated genome summary table")

def generate_wide_matrices(all_results, output_dir, all_genome_names):
    """Generate wide-format matrices"""
    if len(all_results) == 0:
        log_message("WARNING: No results to process")
        return None, None, None
    
    # Matrix 1: GH Family
    gh_matrix = all_results.groupby(['Genome', 'GH_family']).size().unstack(fill_value=0)
    
    # Matrix 2: Enzyme
    enzyme_matrix = all_results.groupby(['Genome', 'Enzyme']).size().unstack(fill_value=0)
    
    # Matrix 3: Cluster
    cluster_matrix = all_results.groupby(['Genome', 'Assigned_cluster']).size().unstack(fill_value=0)
    
    # Add missing genomes with zeros
    if all_genome_names:
        genomes_with_annotations = set(all_results['Genome'].unique())
        missing_genomes = set(all_genome_names) - genomes_with_annotations
        
        if missing_genomes:
            log_message(f"Adding {len(missing_genomes)} genomes with no annotations to matrices (with zeros)")
            
            for genome in missing_genomes:
                for col in gh_matrix.columns:
                    gh_matrix.loc[genome, col] = 0
                for col in enzyme_matrix.columns:
                    enzyme_matrix.loc[genome, col] = 0
                for col in cluster_matrix.columns:
                    cluster_matrix.loc[genome, col] = 0
    
    # Save matrices
    gh_matrix.to_csv(os.path.join(output_dir, 'bifidoAnnotator_tables', 'gh_family_matrix.tsv'), sep='\t')
    enzyme_matrix.to_csv(os.path.join(output_dir, 'bifidoAnnotator_tables', 'enzyme_matrix.tsv'), sep='\t')
    cluster_matrix.to_csv(os.path.join(output_dir, 'bifidoAnnotator_tables', 'cluster_matrix.tsv'), sep='\t')
    
    log_message("Generated wide-format matrices")
    return gh_matrix, enzyme_matrix, cluster_matrix


class HeatmapGenerator:
    """Advanced heatmap generator with improved positioning and color handling"""
    
    def __init__(self, output_dir, annotations_file=None, heatmap_col='blue'):
        """Initialize with output directory and optional annotations file"""
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / 'bifidoAnnotator_tables'
        self.vis_dir = self.output_dir / 'bifidoAnnotator_visualizations'
        self.annotations_file = annotations_file
        self.heatmap_col = heatmap_col
        
        # Matrix file paths
        self.gh_matrix_file = self.tables_dir / 'gh_family_matrix.tsv'
        self.cluster_matrix_file = self.tables_dir / 'cluster_matrix.tsv' 
        self.enzyme_matrix_file = self.tables_dir / 'enzyme_matrix.tsv'
        
        # Data containers
        self.gh_matrix = None
        self.cluster_matrix = None
        self.enzyme_matrix = None
        self.annotations = None
        self.annotation_columns = []
        
        # Color palettes
        self.annotation_colors = {}
        
        # Scaling parameters
        self.scale_factor = 1.0
        self.font_scale = 1.0
        
        # Global color scale
        self.global_vmin = 0
        self.global_vmax = 1
        self.global_cmap = None
        self.global_discrete_levels = [0, 1]
    
    def calculate_adaptive_figsize(self, n_genomes, n_features, has_annotations=False):
        """Calculate optimal figure size"""
        # Base width calculation
        if n_genomes <= 20:
            base_width = 8 + (n_genomes * 0.15)
        elif n_genomes <= 50:  
            base_width = 11 + ((n_genomes-20) * 0.2)
        elif n_genomes <= 150:
            base_width = 17 + ((n_genomes-50) * 0.08)
        else:
            base_width = 25 + ((n_genomes-150) * 0.03)
        
        # Base height calculation
        if n_features <= 10:
            base_height = 6 + (n_features * 0.3)     
        elif n_features <= 50:
            base_height = 9 + ((n_features-10) * 0.22) 
        elif n_features <= 150: 
            base_height = 18 + ((n_features-50) * 0.10) 
        else:
            base_height = 28 + ((n_features-150) * 0.04)
        
        # Adjustments for annotations
        if has_annotations and hasattr(self, 'annotation_columns'):
            n_annotation_cols = len(self.annotation_columns)
            base_height += (n_annotation_cols * 0.8)
            base_width += 2
        
        # Apply bounds
        width = max(8, min(35, base_width))
        height = max(6, min(35, base_height))
        
        return (int(width), int(height))
    
    def determine_figsize(self, matrix_name, manual_figsize=None):
        """Determine figure size"""
        if manual_figsize is not None:
            return tuple(manual_figsize)
        
        if matrix_name == 'gh':
            matrix = self.gh_matrix
        elif matrix_name == 'cluster': 
            matrix = self.cluster_matrix
        elif matrix_name == 'enzyme':
            matrix = self.enzyme_matrix
        else:
            return (12, 8)
        
        n_genomes = len(matrix.index)
        non_zero_features = (matrix.sum(axis=0) > 0).sum()
        n_features = max(1, non_zero_features)
        has_annotations = self.annotations is not None
        
        calculated_size = self.calculate_adaptive_figsize(n_genomes, n_features, has_annotations)
        
        if matrix_name == 'cluster':
            calculated_size = (calculated_size[0], int(calculated_size[1] * 1.3))
            log_message(f"  Cluster heatmap: increased height to {calculated_size[1]} inches", print_also=False)
        
        size_type = "with annotations" if has_annotations else "basic"
        log_message(f"  Auto-calculated size {calculated_size[0]}×{calculated_size[1]} inches ({n_genomes} genomes, {n_features} features, {size_type})", print_also=False)
        
        return calculated_size
    
    def load_data(self):
        """Load and validate all data files"""
        log_message("Loading matrix files for visualization...")
        
        if not all([self.gh_matrix_file.exists(), self.cluster_matrix_file.exists(), self.enzyme_matrix_file.exists()]):
            log_message("ERROR: Matrix files not found.")
            return False
        
        try:
            self.gh_matrix = pd.read_csv(self.gh_matrix_file, sep='\t', index_col=0)
            log_message(f"  GH matrix: {self.gh_matrix.shape}", print_also=False)
        except Exception as e:
            log_message(f"Error loading GH matrix: {e}")
            return False
        
        try:
            self.cluster_matrix = pd.read_csv(self.cluster_matrix_file, sep='\t', index_col=0)
            log_message(f"  Cluster matrix: {self.cluster_matrix.shape}", print_also=False)
        except Exception as e:
            log_message(f"Error loading cluster matrix: {e}")
            return False
        
        try:
            self.enzyme_matrix = pd.read_csv(self.enzyme_matrix_file, sep='\t', index_col=0)
            log_message(f"  Enzyme matrix: {self.enzyme_matrix.shape}", print_also=False)
        except Exception as e:
            log_message(f"Error loading enzyme matrix: {e}")
            return False
        
        # Load annotations if provided
        if self.annotations_file:
            try:
                self.annotations = pd.read_csv(self.annotations_file, sep='\t', index_col=0)
                self.annotation_columns = list(self.annotations.columns)
                log_message(f"  Annotations: {self.annotations.shape}, columns: {self.annotation_columns}", print_also=False)
                
                # Clean species names
                for col in self.annotation_columns:
                    if 'species' in col.lower():
                        self.annotations[col] = self.annotations[col].astype(str).str.replace('s__', '', regex=False)
                        log_message(f"  Cleaned s__ prefix from {col} column", print_also=False)
                
            except Exception as e:
                log_message(f"Warning: Failed to load annotations file: {e}")
                log_message("Proceeding with basic heatmaps without annotations")
                self.annotations = None
                self.annotation_columns = []
        else:
            log_message("  No annotations file provided - generating basic heatmaps")
            self.annotations = None
            self.annotation_columns = []
        
        return self.validate_data()
    
    def validate_data(self):
        """Validate genome consistency"""
        log_message("Validating data consistency...", print_also=False)
        
        gh_genomes = set(self.gh_matrix.index)
        cluster_genomes = set(self.cluster_matrix.index)
        enzyme_genomes = set(self.enzyme_matrix.index)
        
        common_genomes = gh_genomes & cluster_genomes & enzyme_genomes
        
        if self.annotations is not None:
            annotation_genomes = set(self.annotations.index)
            common_genomes = common_genomes & annotation_genomes
            log_message(f"  Annotation genomes: {len(annotation_genomes)}", print_also=False)
        
        if len(common_genomes) == 0:
            log_message("ERROR: No common genomes found!")
            return False
        
        # Filter to common genomes
        common_genomes_sorted = sorted(common_genomes)
        self.gh_matrix = self.gh_matrix.loc[common_genomes_sorted]
        self.cluster_matrix = self.cluster_matrix.loc[common_genomes_sorted]
        self.enzyme_matrix = self.enzyme_matrix.loc[common_genomes_sorted]
        
        if self.annotations is not None:
            self.annotations = self.annotations.loc[common_genomes_sorted]
        
        log_message(f"Using {len(common_genomes)} common genomes for visualization", print_also=False)
        return True
    
    def calculate_color_scale_for_data(self, data_matrix):
        """Calculate color scale with adaptive contrast"""
        actual_max = int(data_matrix.max().max())
        vmin = 0
        vmax = actual_max
        
        log_message(f"  Color scale for this heatmap: 0-{vmax}", print_also=False)
        
        discrete_levels = list(range(vmin, vmax + 1))
        
        if vmin == 0 and vmax > 0:
            colors = ['#FFFFFF']
            
            if self.heatmap_col == 'blue':
                blue_palette = [
                    '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6',
                    '#2171b5', '#08519c', '#08306b', '#041d48', '#021434'
                ]
                
                if vmax == 1:
                    colors.append(blue_palette[5])
                    log_message(f"  Adaptive contrast (max=1): high-contrast color", print_also=False)
                elif vmax == 2:
                    colors.extend([blue_palette[2], blue_palette[6]])
                elif vmax == 3:
                    colors.extend([blue_palette[1], blue_palette[4], blue_palette[7]])
                elif vmax == 4:
                    colors.extend([blue_palette[1], blue_palette[3], blue_palette[5], blue_palette[8]])
                elif vmax == 5:
                    colors.extend([blue_palette[0], blue_palette[2], blue_palette[4], blue_palette[6], blue_palette[8]])
                elif vmax <= 10:
                    colors.extend(blue_palette[:vmax])
                else:
                    colors.extend(blue_palette)
                    for i in range(11, vmax + 1):
                        t = (i - 10) / max(1, vmax - 10)
                        r = int(0x02 * (1 - t * 0.5))
                        g = int(0x14 * (1 - t * 0.5))
                        b = int(0x34 * (1 - t * 0.3))
                        colors.append(f'#{r:02x}{g:02x}{b:02x}')
                
                cmap = ListedColormap(colors, name=f'custom_blue_scale_{vmax}')
                
            else:  # red
                red_palette = [
                    '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d',
                    '#a50f15', '#67000d', '#4a0009', '#330006'
                ]
                
                if vmax == 1:
                    colors.append(red_palette[4])
                elif vmax == 2:
                    colors.extend([red_palette[2], red_palette[5]])
                elif vmax == 3:
                    colors.extend([red_palette[1], red_palette[3], red_palette[6]])
                elif vmax == 4:
                    colors.extend([red_palette[1], red_palette[3], red_palette[5], red_palette[7]])
                elif vmax == 5:
                    colors.extend([red_palette[0], red_palette[2], red_palette[4], red_palette[6], red_palette[8]])
                elif vmax <= 9:
                    colors.extend(red_palette[:vmax])
                else:
                    colors.extend(red_palette)
                    for i in range(10, vmax + 1):
                        t = (i - 9) / max(1, vmax - 9)
                        r = int(0x33 * (1 - t * 0.3))
                        colors.append(f'#{r:02x}0006')
                
                cmap = ListedColormap(colors, name=f'custom_red_scale_{vmax}')
            
            return cmap, vmin, vmax, discrete_levels
        else:
            return 'Reds' if self.heatmap_col == 'red' else 'Blues', 0, 1, [0, 1]
    
    def setup_annotation_colors(self):
        """Create color mapping with distinct palettes - ColorBrewer Dark2 for top bars"""
        if self.annotations is None or len(self.annotation_columns) == 0:
            return
            
        log_message(f"Setting up distinct color palettes for {len(self.annotation_columns)} annotation columns (scheme: {self.heatmap_col})...", print_also=False)
        
        # ColorBrewer Dark2 - used for BOTH blue and red schemes as top bar palette
        dark2_palette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', 
                         '#66a61e', '#e6ab02', '#a6761d', '#666666']
        
        # Define Set3-like palette colors
        set3_colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', 
                       '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']
        
        # Both schemes now use the same palette structure
        color_palettes = [
            # Palette 1: ColorBrewer Dark2 (8 colors) - for top bars
            dark2_palette,
            
            # Palette 2: Backup palette (cycling)
            ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'],
            
            # Palette 3: Grayscale with MAXIMUM contrast (9 colors) - RESERVED for SECOND-TO-LAST bar
            ['#404040', '#595959', '#737373', '#8c8c8c', '#a6a6a6', '#bfbfbf', '#d9d9d9', '#e6e6e6', '#f0f0f0'],
            
            # Palette 4: Custom Set3-like palette (9 colors) - RESERVED for BOTTOM bar
            set3_colors
        ]
        
        n_cols = len(self.annotation_columns)
        
        for i, col in enumerate(self.annotation_columns):
            # Handle missing values
            self.annotations[col] = self.annotations[col].fillna('N.A.')
            
            unique_values = sorted(self.annotations[col].unique())
            n_values = len(unique_values)
            n_non_na_values = len([v for v in unique_values if v != 'N.A.'])
            
            log_message(f"  {col}: {n_values} unique values", print_also=False)
            
            # BOTTOM BAR (closest to heatmap) - gets Set3-like palette
            if i == n_cols - 1:
                palette_idx = len(color_palettes) - 1  # Last palette (index 3)
                chosen_palette = color_palettes[palette_idx]
                is_dark2_palette = False
                log_message(f"    Assigning CUSTOM SET3-LIKE to '{col}' (bottom bar, closest to heatmap)", print_also=False)
            
            # SECOND-TO-LAST BAR - gets grayscale
            elif i == n_cols - 2:
                palette_idx = len(color_palettes) - 2  # Second-to-last palette (index 2 - grayscale)
                chosen_palette = color_palettes[palette_idx]
                is_dark2_palette = False
                log_message(f"    Assigning GRAYSCALE to '{col}' (second-to-last bar)", print_also=False)
            
            # OTHER BARS (top bars) - use Dark2 or cycle
            else:
                # Other columns cycle through Dark2 and backup palette (excluding grayscale and bottom palette)
                available_palettes = len(color_palettes) - 2  # Exclude last 2 (grayscale and bottom)
                palette_idx = i % available_palettes
                chosen_palette = color_palettes[palette_idx]
                is_dark2_palette = (palette_idx == 0)  # Dark2 is always palette 0 now
                
                if is_dark2_palette:
                    log_message(f"    Using ColorBrewer Dark2 for '{col}'", print_also=False)
                else:
                    log_message(f"    Using backup palette {palette_idx + 1} for '{col}'", print_also=False)
            
            # Create color mapping with maximum divergence
            color_dict = {}
            
            # Select maximally divergent color indices from palette
            palette_size = len(chosen_palette)
            if n_non_na_values <= palette_size:
                if is_dark2_palette:
                    # Special handling for Dark2 palette - qualitative, not diverging
                    # Dark2 colors: [0]Teal, [1]Orange, [2]Purple, [3]Magenta, [4]Olive, [5]Yellow, [6]Brown, [7]Gray
                    if n_non_na_values == 1:
                        color_indices = [0]  # Teal
                    elif n_non_na_values == 2:
                        color_indices = [0, 1]  # Teal, Orange (cool vs warm)
                    elif n_non_na_values == 3:
                        color_indices = [0, 1, 2]  # Teal, Orange, Purple (cool, warm, mid)
                    elif n_non_na_values == 4:
                        # CRITICAL: Maximum color differentiation across spectrum
                        # Avoid similar colors (no two greens/teals, no two yellows/oranges)
                        # Avoid Set3 overlap: Set3 has teal(#8dd3c7), yellow(#ffffb3, #fdb462)
                        # Choose: Teal(different shade), Orange, Magenta(pink), Olive(green)
                        # [0]Teal=#1b9e77 (darker than Set3's #8dd3c7) ✓
                        # [1]Orange=#d95f02 (reddish-orange, distinct from Set3's peachy #fdb462) ✓
                        # [3]Magenta=#e7298a (hot pink, distinct from Set3's pale pink #fccde5) ✓
                        # [4]Olive=#66a61e (yellow-green, distinct from Set3's light green #b3de69) ✓
                        color_indices = [0, 1, 3, 4]  # Teal, Orange, Magenta, Olive
                        log_message(f"    Dark2 with 4 values: using maximally different colors", print_also=False)
                        log_message(f"      [0]Teal=#1b9e77, [1]Orange=#d95f02, [3]Magenta=#e7298a, [4]Olive=#66a61e", print_also=False)
                        log_message(f"      Avoiding overlap with Set3 pastels", print_also=False)
                    elif n_non_na_values == 5:
                        color_indices = [0, 1, 2, 3, 4]  # Teal, Orange, Purple, Magenta, Olive
                    elif n_non_na_values == 6:
                        color_indices = [0, 1, 2, 3, 4, 5]  # Add Yellow
                    elif n_non_na_values == 7:
                        color_indices = [0, 1, 2, 3, 4, 5, 6]  # Add Brown
                    else:  # 8 values
                        color_indices = list(range(8))  # Use all colors
                    
                    log_message(f"    DARK2 qualitative palette: using indices {color_indices}", print_also=False)
                    
                else:
                    # For NON-DARK2 palettes: special handling for grayscale and Set3
                    if palette_idx == 2:  # Grayscale
                        if n_non_na_values == 1:
                            color_indices = [4]  # Middle
                        elif n_non_na_values == 2:
                            color_indices = [1, 8]  # Dark and very light
                        elif n_non_na_values == 3:
                            color_indices = [0, 4, 8]  # Dark, Medium, Light
                        elif n_non_na_values == 4:
                            color_indices = [0, 3, 6, 8]  # Maximum spread
                        elif n_non_na_values == 5:
                            color_indices = [0, 2, 4, 6, 8]
                        else:
                            # Evenly distribute
                            step = (palette_size - 1) / (n_non_na_values - 1)
                            color_indices = [int(round(j * step)) for j in range(n_non_na_values)]
                        log_message(f"    GRAYSCALE palette: using indices {color_indices}", print_also=False)
                    
                    elif palette_idx == 3:  # Set3-like (bottom bar)
                        # Sequential selection for Set3
                        color_indices = list(range(n_non_na_values))
                        log_message(f"    SET3-LIKE palette: using indices {color_indices}", print_also=False)
                    
                    else:  # Backup palette (index 1)
                        color_indices = list(range(n_non_na_values))
                        log_message(f"    BACKUP palette: using indices {color_indices}", print_also=False)
                
                log_message(f"    Selected {n_non_na_values} colors from palette", print_also=False)
            else:
                # More values than colors in palette - cycle through
                color_indices = list(range(n_non_na_values))
                log_message(f"    More values than palette colors - cycling through", print_also=False)
            
            # Assign colors to values
            color_idx = 0
            for value in unique_values:
                if value == 'N.A.':
                    color_dict[value] = '#F0F0F0'  # Light gray for N.A.
                    log_message(f"    N.A.: #F0F0F0 (missing values)", print_also=False)
                else:
                    if n_non_na_values > 0:
                        palette_idx_to_use = color_indices[color_idx] % palette_size
                        color_dict[value] = chosen_palette[palette_idx_to_use]
                        log_message(f"    {value}: {color_dict[value]} (index {palette_idx_to_use})", print_also=False)
                        color_idx += 1
                    else:
                        color_dict[value] = '#CCCCCC'
            
            self.annotation_colors[col] = color_dict
        
        log_message("Distinct color palettes assigned", print_also=False)
    
    def calculate_dynamic_positions(self, heatmap_ax, fig, n_annotation_rows, show_column_labels, n_features):
        """Calculate positions with no overlaps"""
        fig.canvas.draw()
        
        hm_pos = heatmap_ax.get_position()
        hm_left, hm_bottom, hm_width, hm_height = hm_pos.x0, hm_pos.y0, hm_pos.width, hm_pos.height
        hm_right = hm_left + hm_width
        
        log_message(f"  Heatmap data area: left={hm_left:.3f}, bottom={hm_bottom:.3f}, width={hm_width:.3f}, height={hm_height:.3f}", print_also=False)
        
        cbar_width = 0.12   
        cbar_height = 0.025 
        cbar_left = hm_right + 0.02   
        cbar_bottom = hm_bottom - 0.05  
        
        if show_column_labels:
            if n_features >= 40:
                legend_space_needed = 0.10
            elif n_features >= 20:
                legend_space_needed = 0.14
            else:
                legend_space_needed = 0.18
        else:
            legend_space_needed = 0.06
        
        legend_bottom = hm_bottom - legend_space_needed
        legend_left = hm_left + (hm_width * 0.1)
        
        return {
            'colorbar': (cbar_left, cbar_bottom, cbar_width, cbar_height),
            'legend': (legend_left, legend_bottom),
            'heatmap': (hm_left, hm_bottom, hm_width, hm_height)
        }
    
    def create_clustermap(self, data, title, output_file_base, manual_figsize=None, matrix_name='unknown'):
        """Create hierarchically clustered heatmap"""
        log_message(f"Creating: {title}")
        
        figsize = self.determine_figsize(matrix_name, manual_figsize)
        log_message(f"  Figure size: {figsize[0]}×{figsize[1]} inches")
        
        data_t = data.T
        log_message(f"  Data: {data_t.shape} (features × genomes)", print_also=False)
        
        annotation_color_lists = []
        if self.annotations is not None and len(self.annotation_columns) > 0:
            genome_order = data_t.columns
            
            for col in self.annotation_columns:
                color_list = []
                for genome in genome_order:
                    if genome in self.annotations.index:
                        annotation_value = self.annotations.loc[genome, col]
                        if pd.isna(annotation_value):
                            annotation_value = 'N.A.'
                        color_list.append(self.annotation_colors[col][annotation_value])
                    else:
                        color_list.append('white')
                annotation_color_lists.append(color_list)
        
        non_zero_features = (data_t.sum(axis=1) > 0)
        data_filtered = data_t[non_zero_features]
        log_message(f"  Filtered to {len(data_filtered)} non-zero features", print_also=False)
        
        if len(data_filtered) == 0:
            log_message("WARNING: No non-zero features found")
            return None
        
        n_genomes = len(data_filtered.columns)
        show_column_labels = n_genomes <= 150
        log_message(f"  Genomes: {n_genomes}, Show labels: {show_column_labels}", print_also=False)
        
        cmap, vmin, vmax, discrete_levels = self.calculate_color_scale_for_data(data_filtered)
        boundaries = np.arange(vmin, vmax + 2) - 0.5  
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        
        plt.figure(figsize=figsize)
        
        dendrogram_ratio = 0.15  
        
        if annotation_color_lists:
            n_annotation_bars = len(annotation_color_lists)
            
            if n_genomes < 15:
                base_ratio = 0.035
            elif n_genomes < 30:
                base_ratio = 0.025
            elif n_genomes < 100:
                base_ratio = 0.018
            else:
                base_ratio = 0.012
            
            colors_ratio = base_ratio * n_annotation_bars
            
            if matrix_name == 'cluster':
                colors_ratio = (colors_ratio / 1.3) * 0.85
                colors_ratio = max(0.035, colors_ratio)
            else:
                colors_ratio = max(0.03, min(0.20, colors_ratio))
            
            log_message(f"  Annotation bars: {n_annotation_bars} bars at ratio {colors_ratio:.3f}", print_also=False)
        else:
            colors_ratio = 0
        
        if annotation_color_lists:
            g = sns.clustermap(
                data_filtered,
                method='average',
                metric='euclidean',
                cmap=cmap,
                norm=norm,  
                figsize=figsize,
                dendrogram_ratio=dendrogram_ratio,
                colors_ratio=colors_ratio,
                col_colors=annotation_color_lists,
                linewidths=1.5,
                linecolor='black',
                xticklabels=show_column_labels,
                yticklabels=True,
                tree_kws={'linewidths': 1.5}
            )
        else:
            g = sns.clustermap(
                data_filtered,
                method='average',
                metric='euclidean',
                cmap=cmap,
                norm=norm,  
                figsize=figsize,
                dendrogram_ratio=dendrogram_ratio,
                linewidths=1.5,
                linecolor='black',
                xticklabels=show_column_labels,
                yticklabels=True,
                tree_kws={'linewidths': 1.5}
            )
        
        if hasattr(g, 'ax_cbar') and g.ax_cbar is not None:
            g.ax_cbar.remove()
        
        if show_column_labels:
            bottom_space = 0.30 + (len(annotation_color_lists) * 0.02)  
        else:
            bottom_space = 0.20 + (len(annotation_color_lists) * 0.02)
        
        bottom_space = max(bottom_space, 0.25)
        
        if matrix_name == 'cluster':
            top_space = 0.92 - (len(annotation_color_lists) * 0.015)
        else:
            top_space = 0.85 - (len(annotation_color_lists) * 0.02)
        
        plt.subplots_adjust(
            left=0.12,
            right=0.75,
            top=top_space,
            bottom=bottom_space
        )
        
        g.fig.canvas.draw()
        
        n_annotation_rows = len(annotation_color_lists) if annotation_color_lists else 0
        positions = self.calculate_dynamic_positions(g.ax_heatmap, g.fig, n_annotation_rows, show_column_labels, len(data_filtered))
        
        g.ax_heatmap.set_xlabel('Genomes', fontsize=10, fontweight='bold', fontfamily='Arial')
        g.ax_heatmap.set_ylabel('Features', fontsize=10, fontweight='bold', fontfamily='Arial')
        
        if show_column_labels:
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8, fontfamily='Arial')
        
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8, fontfamily='Arial')
        
        if annotation_color_lists:
            self.add_annotation_labels(g.fig, g)
            self.create_annotation_separation(g)
            self.add_multiple_legends(g.fig, positions['legend'], figsize)
        
        self.add_dynamic_colorbar(g.fig, positions['colorbar'], cmap, vmin, vmax, discrete_levels)
        
        png_file = f"{output_file_base}.png"
        pdf_file = f"{output_file_base}.pdf"
        
        plt.savefig(png_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='png')
        plt.savefig(pdf_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='pdf')
        
        log_message(f"  Saved: {Path(png_file).name} and {Path(pdf_file).name}")
        plt.close()
        
        return g
    
    def add_dynamic_colorbar(self, fig, position, cmap, vmin, vmax, discrete_levels):
        """Add horizontal colorbar"""
        cbar_left, cbar_bottom, cbar_width, cbar_height = position
        
        axes_to_remove = []
        for ax in fig.get_axes():
            pos = ax.get_position()
            if (pos.width < 0.2 and pos.x0 > 0.65) or (pos.height < 0.05):
                axes_to_remove.append(ax)
        
        for ax in axes_to_remove:
            ax.remove()
        
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        
        boundaries = np.arange(vmin, vmax + 2) - 0.5
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        
        cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
        
        cbar.set_label('Copy Number', rotation=0, labelpad=8, 
                      fontsize=9, fontfamily='Arial', ha='center')
        
        max_ticks = min(8, len(discrete_levels))
        if len(discrete_levels) <= max_ticks:
            cbar.set_ticks(discrete_levels)
            cbar.set_ticklabels([str(int(x)) for x in discrete_levels])
        else:
            step = max(1, len(discrete_levels) // (max_ticks - 2))
            subset_levels = [discrete_levels[0]]
            subset_levels.extend(discrete_levels[step::step])
            if discrete_levels[-1] not in subset_levels:
                subset_levels.append(discrete_levels[-1])
            subset_levels = sorted(list(set(subset_levels)))
            cbar.set_ticks(subset_levels)
            cbar.set_ticklabels([str(int(x)) for x in subset_levels])
        
        cbar.ax.tick_params(labelsize=8, length=3, pad=2)
        
        for label in cbar.ax.get_xticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(8)
        
        cbar_ax.patch.set_facecolor('white')
        cbar_ax.patch.set_alpha(1.0)
        
        return cbar
    
    def add_multiple_legends(self, fig, position, figsize):
        """Add multiple legends"""
        legend_left, legend_bottom = position
        n_annotations = len(self.annotation_columns)
        
        legend_font_size = max(8, min(11, figsize[1] * 0.4))  
        
        legend_widths = []
        for col in self.annotation_columns:
            unique_values = sorted(self.annotation_colors[col].keys())
            max_text_length = max(len(str(v)) for v in unique_values)
            n_values = len(unique_values)
            
            if n_values <= 4:
                ncol = 1
                est_width = max(0.15, max_text_length * 0.008)  
            elif n_values <= 8:
                ncol = 2
                est_width = max(0.22, max_text_length * 0.010)  
            else:
                ncol = min(3, (n_values + 2) // 3)
                est_width = max(0.28, max_text_length * 0.012)  
            
            legend_widths.append((est_width, ncol, n_values))
        
        total_width = sum(w[0] for w in legend_widths)
        spacing = 0.06  
        total_width_with_spacing = total_width + (spacing * max(0, n_annotations - 1))
        
        heatmap_center = legend_left + (0.6 * 0.5)  
        start_x = heatmap_center - (total_width_with_spacing / 2)
        current_x = max(legend_left, start_x)  
        
        for i, col in enumerate(self.annotation_columns):
            est_width, ncol, n_values = legend_widths[i]
            
            legend_patches = []
            unique_values = sorted(self.annotation_colors[col].keys())
            
            for value in unique_values:
                color = self.annotation_colors[col][value]
                display_name = str(value)
                if len(display_name) > 20:
                    display_name = display_name[:17] + "..."
                
                patch = mpatches.Patch(color=color, label=display_name, 
                                     edgecolor='black', linewidth=0.8)
                legend_patches.append(patch)
            
            legend = fig.legend(handles=legend_patches,
                              title=col,
                              bbox_to_anchor=(current_x, legend_bottom),
                              bbox_transform=fig.transFigure,
                              loc='upper left',
                              ncol=ncol,
                              fontsize=legend_font_size,
                              title_fontsize=legend_font_size,
                              frameon=False,
                              columnspacing=0.4,      
                              handletextpad=0.2,      
                              handlelength=0.8,       
                              borderaxespad=0,
                              prop={'family': 'Arial', 'size': legend_font_size})
            
            legend.get_title().set_fontfamily('Arial')
            legend.get_title().set_fontweight('bold')
            
            current_x += est_width + spacing
    
    def create_annotation_separation(self, clustermap_obj):
        """Create visual separation"""
        separation_gap = 0.006
        
        heatmap_pos = clustermap_obj.ax_heatmap.get_position()
        
        if hasattr(clustermap_obj, 'ax_col_colors') and clustermap_obj.ax_col_colors is not None:
            col_colors_pos = clustermap_obj.ax_col_colors.get_position()
            
            new_heatmap_top = col_colors_pos.y0 - separation_gap
            new_heatmap_bottom = new_heatmap_top - heatmap_pos.height
            
            new_heatmap_pos = [heatmap_pos.x0, new_heatmap_bottom, heatmap_pos.width, heatmap_pos.height]
            clustermap_obj.ax_heatmap.set_position(new_heatmap_pos)
            
            if hasattr(clustermap_obj, 'ax_row_dendrogram') and clustermap_obj.ax_row_dendrogram is not None:
                row_dend_pos = clustermap_obj.ax_row_dendrogram.get_position()
                new_row_dend_pos = [row_dend_pos.x0, new_heatmap_bottom, row_dend_pos.width, heatmap_pos.height]
                clustermap_obj.ax_row_dendrogram.set_position(new_row_dend_pos)
            
            clustermap_obj.fig.canvas.draw_idle()
    
    def add_annotation_labels(self, fig, clustermap_obj):
        """Add labels for each annotation row"""
        if hasattr(clustermap_obj, 'ax_col_colors') and clustermap_obj.ax_col_colors is not None:
            col_colors_pos = clustermap_obj.ax_col_colors.get_position()
            
            n_annotations = len(self.annotation_columns)
            row_height = col_colors_pos.height / n_annotations
            
            for i, col in enumerate(self.annotation_columns):
                label_x = col_colors_pos.x0 + col_colors_pos.width + 0.01
                label_y = col_colors_pos.y0 + col_colors_pos.height - (i + 0.5) * row_height
                
                fig.text(label_x, label_y, col,
                        rotation=0,
                        verticalalignment='center',
                        horizontalalignment='left',
                        fontsize=11,
                        fontweight='bold',
                        fontfamily='Arial')
    
    def generate_heatmaps(self, gh_figsize=None, cluster_figsize=None, enzyme_figsize=None):
        """Generate all heatmaps"""
        log_message("Generating publication-ready heatmaps...")
        
        if not self.load_data():
            log_message("ERROR: Failed to load data")
            return False
        
        if self.annotations is not None:
            self.setup_annotation_colors()
        
        annotation_info = f" with {len(self.annotation_columns)} annotation columns" if self.annotations is not None else " (basic heatmaps)"
        log_message(f"Creating heatmaps{annotation_info} - {self.heatmap_col} color scheme")
        
        try:
            gh_output_base = self.vis_dir / 'gh_family_heatmap'
            self.create_clustermap(
                self.gh_matrix,
                'GH Family Abundance Heatmap',
                str(gh_output_base),
                manual_figsize=gh_figsize,
                matrix_name='gh'
            )
        except Exception as e:
            log_message(f"ERROR: Failed to create GH heatmap: {e}")
        
        try:
            cluster_output_base = self.vis_dir / 'cluster_heatmap'
            self.create_clustermap(
                self.cluster_matrix,
                'Cluster Abundance Heatmap',
                str(cluster_output_base),
                manual_figsize=cluster_figsize,
                matrix_name='cluster'
            )
        except Exception as e:
            log_message(f"ERROR: Failed to create cluster heatmap: {e}")
        
        try:
            enzyme_output_base = self.vis_dir / 'enzyme_heatmap'
            self.create_clustermap(
                self.enzyme_matrix,
                'Enzyme Abundance Heatmap',
                str(enzyme_output_base),
                manual_figsize=enzyme_figsize,
                matrix_name='enzyme'
            )
        except Exception as e:
            log_message(f"ERROR: Failed to create enzyme heatmap: {e}")
        
        log_message(f"Heatmap generation completed! Files saved in: {self.vis_dir}")
        return True


def main():
    """Main function"""
    print("=" * 80)
    print("Combined bifidoAnnotator: Complete GH Annotation & Visualization Pipeline")
    print("=" * 80)
    
    # Parse arguments FIRST
    args = parse_arguments()
    
    # Check dependencies
    check_dependencies()
    
    # Validate arguments
    if args.genome_directory and not args.sample_file:
        print("ERROR: --sample_file is required when using --genome_directory")
        sys.exit(1)
    
    # Create output structure
    create_output_structure(args.output_dir)
    
    # Initialize logging
    initialize_log(args.output_dir, args)
    
    # Load mapping file
    mapping_df = load_mapping_file(args.mapping_file)
    
    # Log input processing
    log_section("INPUT PROCESSING")
    
    # Determine input files
    input_files = []
    genome_names = []
    
    if args.input_file:
        input_files.append(args.input_file)
        genome_names.append(Path(args.input_file).stem)
        log_message(f"Single file mode: {args.input_file}", print_also=False)
    else:
        log_message(f"Batch mode: reading genome list from {args.sample_file}", print_also=False)
        with open(args.sample_file, 'r') as f:
            for line in f:
                genome_name = line.strip()
                if genome_name:
                    for ext in ['.fasta', '.fa', '.faa']:
                        fasta_path = os.path.join(args.genome_directory, f"{genome_name}{ext}")
                        if os.path.exists(fasta_path):
                            input_files.append(fasta_path)
                            genome_names.append(genome_name)
                            log_message(f"Found: {fasta_path}", print_also=False)
                            break
                    else:
                        log_message(f"WARNING: No FASTA file found for {genome_name}")
    
    if not input_files:
        log_message("ERROR: No valid input files found")
        sys.exit(1)
    
    log_message(f"Processing {len(input_files)} genome(s)")
    
    # Log genome processing
    log_section("GENOME PROCESSING")
    
    # Process each genome
    all_results = []
    
    for input_file, genome_name in zip(input_files, genome_names):
        log_message(f"\nProcessing: {genome_name}")
        log_message(f"Input file: {input_file}", print_also=False)
        
        file_size = os.path.getsize(input_file) / 1024
        log_message(f"File size: {file_size:.1f} KB", print_also=False)
        
        output_prefix = os.path.join(args.output_dir, f"{genome_name}")
        success = run_mmseqs_search(input_file, args.bifdb, output_prefix, 
                                  args.threads, args.sensitivity)
        
        if success:
            results_file = f"{output_prefix}_results.tsv"
            genome_results = process_mmseqs_results(results_file, mapping_df, genome_name)
            
            if not genome_results.empty:
                all_results.append(genome_results)
            
            if os.path.exists(results_file):
                os.remove(results_file)
        else:
            log_message(f"Failed to process {genome_name}")
    
    # Log output generation
    log_section("OUTPUT GENERATION")
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        log_message("\nGenerating output files...")
        generate_detailed_annotations(combined_results, args.output_dir)
        generate_genome_summary(combined_results, args.output_dir, genome_names)
        matrices = generate_wide_matrices(combined_results, args.output_dir, genome_names)
        
        # Generate heatmaps
        log_section("VISUALIZATION GENERATION")
        heatmap_generator = HeatmapGenerator(args.output_dir, args.annotations_file, args.heatmap_col)
        heatmap_success = heatmap_generator.generate_heatmaps(
            gh_figsize=args.gh_figsize,
            cluster_figsize=args.cluster_figsize,
            enzyme_figsize=args.enzyme_figsize
        )
        
        if heatmap_success:
            log_message("Advanced heatmap generation completed successfully")
        else:
            log_message("WARNING: Heatmap generation encountered errors")
        
        log_message(f"\nAnalysis complete! Results saved in: {args.output_dir}")
        log_message(f"Total annotations: {len(combined_results)}")
        log_message(f"Unique genomes: {combined_results['Genome'].nunique()}")
        log_message(f"GH families detected: {combined_results['GH_family'].nunique()}")
        
        finalize_log(combined_results, genome_names, matrices)
        
    else:
        log_message("\nWARNING: No annotations found across all input genomes")
        log_message("No output files generated.")
        
        finalize_log(pd.DataFrame(), genome_names, (None, None, None))
    
    print("=" * 80)
    print(f"Complete log saved to: {os.path.join(args.output_dir, 'bifidoAnnotator_log.txt')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
