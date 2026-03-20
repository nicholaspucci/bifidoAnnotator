#!/usr/bin/env python3
"""
Combined bifidoAnnotator: Complete pipeline for the annotation of bifidobacterial enzymes
involved in HMG-utilization and publication-ready heatmap generation with adaptive sizing.

This tool performs hierarchical annotation of GH-encoding genes using MMseqs2,
generates data matrices, and creates sophisticated heatmaps with automatic size optimization.

Author: Nicholas Pucci & Daniel R. Mende
Version: 2.0 - Transporter module, HMG-utilization row annotation bar,
                cluster row filtering, updated color palettes
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
TRANSPORTER_IDENTITY_THRESHOLD = 80.0

# Database download configuration
DEFAULT_DB_DIR = os.path.expanduser('~/.bifidoannotator/database')
ZENODO_URL = "https://zenodo.org/records/19133752/files/bifDB_dir.tar.gz"
ZENODO_MD5 = "0a63c9ca94c0666f71041970ee126676"


# =============================================================================
# AUTO-DOWNLOAD
# =============================================================================

def download_database():
    """Download and extract the GH reference database from Zenodo on first use."""
    db_path = os.path.join(DEFAULT_DB_DIR, 'bifDB_dir', 'bifDB')
    marker_file = os.path.join(DEFAULT_DB_DIR, '.downloaded')

    # Check if already downloaded
    if os.path.exists(marker_file) and os.path.exists(db_path):
        return True

    print("=" * 80)
    print("FIRST RUN: Downloading reference database from Zenodo")
    print("This is a one-time download (~141 MB compressed, ~1 GB extracted)")
    print(f"Downloading from: {ZENODO_URL}")
    print("=" * 80)

    try:
        os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
        tar_file = os.path.join(DEFAULT_DB_DIR, 'bifDB_dir.tar.gz')

        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                  end='', flush=True)

        urllib.request.urlretrieve(ZENODO_URL, tar_file, reporthook=report_progress)
        print()

        # Verify checksum
        print("Verifying download integrity...")
        md5_hash = hashlib.md5()
        with open(tar_file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)

        if md5_hash.hexdigest() != ZENODO_MD5:
            print("ERROR: Checksum mismatch! Download may be corrupted.")
            os.remove(tar_file)
            return False

        print("Checksum verified!")

        # Extract
        print("Extracting database...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(path=DEFAULT_DB_DIR)

        os.remove(tar_file)

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
    """Get default paths for the GH database, GH mapping file, and transporter mapping file."""
    db_dir = DEFAULT_DB_DIR
    bifdb_path = os.path.join(db_dir, 'bifDB_dir', 'bifDB')
    mapping_path = os.path.join(db_dir, 'mapping_file.tsv')
    transporter_mapping_path = None

    # Try packaged mapping files first (pip installation)
    try:
        import pkg_resources
        package_mapping = pkg_resources.resource_filename(
            'bifidoannotator', 'data/mapping_file.tsv')
        if os.path.exists(package_mapping):
            mapping_path = package_mapping
        package_tp_mapping = pkg_resources.resource_filename(
            'bifidoannotator', 'data/mapping_file_transporters.tsv')
        if os.path.exists(package_tp_mapping):
            transporter_mapping_path = package_tp_mapping
    except Exception:
        pass

    return bifdb_path, mapping_path, transporter_mapping_path


# =============================================================================
# LOGGING
# =============================================================================

def initialize_log(output_dir, args):
    """Initialize the log file."""
    global LOG_FILE, START_TIME
    START_TIME = time.time()

    LOG_FILE = os.path.join(output_dir, 'bifidoAnnotator_log.txt')

    with open(LOG_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMBINED BIFIDOANNOTATOR: Complete GH Annotation & Visualization Pipeline\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command line: {' '.join(sys.argv)}\n\n")

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
        f.write(f"Transporter database: {args.transporter_db if args.transporter_db else 'None (transporter module disabled)'}\n")
        f.write(f"Transporter mapping: {args.transporter_mapping if args.transporter_mapping else 'None'}\n")
        f.write(f"Transporter identity threshold: {TRANSPORTER_IDENTITY_THRESHOLD}%\n")
        f.write(f"GH figure size: {'Auto-adaptive' if not args.gh_figsize else f'{args.gh_figsize[0]}x{args.gh_figsize[1]}'}\n")
        f.write(f"Cluster figure size: {'Auto-adaptive' if not args.cluster_figsize else f'{args.cluster_figsize[0]}x{args.cluster_figsize[1]}'}\n")
        f.write(f"Enzyme figure size: {'Auto-adaptive' if not args.enzyme_figsize else f'{args.enzyme_figsize[0]}x{args.enzyme_figsize[1]}'}\n")
        f.write(f"Heatmap color scheme: {args.heatmap_col}\n")
        f.write(f"Cluster heatmap row filter: {'HMG only (Yes)' if args.hmg_only else ('HMG + Unknown (Yes/Unknown)' if not args.all_genes else 'All genes')}\n\n")


def log_message(message, print_also=True):
    """Write message to log file and optionally print."""
    if print_also:
        print(message)
    if LOG_FILE:
        with open(LOG_FILE, 'a') as f:
            f.write(message + "\n")


def log_section(title):
    """Log a section header."""
    message = f"\n{title.upper()}\n" + "-" * len(title) + "\n"
    log_message(message, print_also=False)


def finalize_log(combined_results, genome_names, matrices,
                 combined_transporter_results=None):
    """Write final statistics to log file."""
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
            f.write("OVERALL ANNOTATION STATISTICS:\n")
            f.write(f"Total genomes processed: {len(genome_names)}\n")
            f.write(f"Genomes with annotations: {combined_results['Genome'].nunique()}\n")
            f.write(f"Genomes without annotations: {len(genome_names) - combined_results['Genome'].nunique()}\n")
            f.write(f"Total sequences annotated: {len(combined_results)}\n")
            f.write(f"Unique GH families detected: {combined_results['GH_family'].nunique()}\n")
            f.write(f"Unique clusters detected: {combined_results['Assigned_cluster'].nunique()}\n\n")

            f.write("GH FAMILY DISTRIBUTION:\n")
            gh_family_counts = combined_results['GH_family'].value_counts()
            for family, count in gh_family_counts.head(10).items():
                percentage = (count / len(combined_results)) * 100
                f.write(f"  {family}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            f.write("VALIDATION STATUS BREAKDOWN:\n")
            validation_counts = combined_results['Validation_status'].value_counts()
            for status, count in validation_counts.items():
                percentage = (count / len(combined_results)) * 100
                f.write(f"  {status}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            f.write("ANNOTATION QUALITY:\n")
            f.write(f"  Mean percent identity: {combined_results['pident'].mean():.1f}%\n")
            f.write(f"  Mean bit score: {combined_results['bits'].mean():.1f}\n")
            f.write(f"  Identity range: {combined_results['pident'].min():.1f}% - {combined_results['pident'].max():.1f}%\n\n")
        else:
            f.write("No annotations found across all genomes.\n\n")

        if combined_transporter_results is not None and len(combined_transporter_results) > 0:
            f.write("TRANSPORTER ANNOTATION STATISTICS:\n")
            f.write(f"  Total transporter sequences annotated: {len(combined_transporter_results)}\n")
            f.write(f"  Genomes with transporter annotations: {combined_transporter_results['Genome'].nunique()}\n")
            f.write(f"  Unique transporter genes detected: {combined_transporter_results['Transporter_gene'].nunique()}\n")
            if 'Transporter_substrate_category' in combined_transporter_results.columns:
                cat_counts = combined_transporter_results['Transporter_substrate_category'].value_counts()
                f.write("  Substrate category distribution:\n")
                for cat, count in cat_counts.items():
                    f.write(f"    {cat}: {count}\n")
            f.write(f"  Mean transporter percent identity: {combined_transporter_results['pident'].mean():.1f}%\n\n")
        elif combined_transporter_results is not None:
            f.write("TRANSPORTER ANNOTATION STATISTICS:\n")
            f.write("  No transporter annotations found.\n\n")

        f.write("OUTPUT FILES GENERATED:\n")
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


# =============================================================================
# MATPLOTLIB DEFAULTS
# =============================================================================

plt.rcParams['font.family'] = ['Nimbus Sans', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 14


# =============================================================================
# DEPENDENCIES & ARGUMENTS
# =============================================================================

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        subprocess.run(['mmseqs', '-h'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("=" * 80)
        print("ERROR: MMseqs2 not found")
        print("=" * 80)
        print("\nMMseqs2 is required but not installed in your PATH.")
        print("\nTo install MMseqs2:")
        print("  • Conda: conda install -c bioconda mmseqs2")
        print("  • Ubuntu/Debian: sudo apt-get install mmseqs2")
        print("  • macOS: brew install mmseqs2")
        print("\nFor more information: https://github.com/soedinglab/MMseqs2")
        print("=" * 80)
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    default_bifdb, default_mapping, default_tp_mapping = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Combined bifidoAnnotator: Complete GH annotation and visualization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single genome (GH database auto-downloaded on first run)
  bifidoAnnotator -i genome.fasta -o results

  # Batch processing
  bifidoAnnotator -d genomes_dir -s sample_list.txt -o results

  # With genome metadata annotations
  bifidoAnnotator -i genome.fasta --annotations_file metadata.tsv -o results

  # With transporter module
  bifidoAnnotator -i genome.fasta --transporter_db /path/to/bifTPDB \\
                  --transporter_mapping /path/to/mapping_file_transporters.tsv -o results

  # Show only HMG-confirmed clusters in cluster heatmap
  bifidoAnnotator -i genome.fasta --hmg-only -o results

  # Using custom database
  bifidoAnnotator -i genome.fasta --bifdb /custom/db \\
                  --mapping_file /custom/mapping.tsv -o results

Note: On first run, the GH reference database (~141 MB) will be automatically
downloaded from Zenodo (DOI: 10.5281/zenodo.19133752). This only happens once.
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input_file',
                             help='Path to single input FASTA file')
    input_group.add_argument('-d', '--genome_directory',
                             help='Path to directory containing input FASTA files')

    parser.add_argument('-s', '--sample_file',
                        help='Text file listing genome names (required with -d)')
    parser.add_argument('-o', '--output_dir', default='bifidoAnnotator_output',
                        help='Output directory (default: bifidoAnnotator_output)')

    # Database arguments — optional, auto-download on first run
    parser.add_argument('--bifdb', default=default_bifdb,
                        help='Path to MMseqs2 GH reference database '
                             '(default: auto-downloaded from Zenodo on first run)')
    parser.add_argument('--mapping_file', default=default_mapping,
                        help='Path to GH mapping file '
                             '(default: packaged or downloaded with database)')

    parser.add_argument('--annotations_file',
                        help='TSV file with genome metadata for heatmap annotation bars')
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
    parser.add_argument('-hc', '--heatmap_col', type=str, default='blue',
                        choices=['red', 'blue'],
                        help='Color scheme for heatmap (default: blue)')

    # Cluster heatmap row filtering (mutually exclusive)
    hmg_filter_group = parser.add_mutually_exclusive_group()
    hmg_filter_group.add_argument('--hmg-unknown', action='store_true', default=False,
                                  help='Show only clusters with HMG-utilization = Yes or Unknown '
                                       '(default behaviour).')
    hmg_filter_group.add_argument('--hmg-only', action='store_true', default=False,
                                  help='Show only clusters with HMG-utilization = Yes.')
    hmg_filter_group.add_argument('--all-genes', action='store_true', default=False,
                                  help='Show all clusters regardless of HMG-utilization status.')

    # Transporter module (optional)
    parser.add_argument('--transporter_db', default=None,
                        help='Path to MMseqs2 transporter reference database (optional)')
    parser.add_argument('--transporter_mapping', default=default_tp_mapping,
                        help='Path to transporter mapping TSV file '
                             '(required if --transporter_db is provided)')

    args = parser.parse_args()

    # Auto-download GH database if not present
    if not os.path.exists(args.bifdb):
        print(f"GH database not found at: {args.bifdb}")
        if not download_database():
            print("\nERROR: Could not download database automatically.")
            print("\nManual download instructions:")
            print(f"1. Download: {ZENODO_URL}")
            print(f"2. Extract to: {DEFAULT_DB_DIR}")
            print(f"3. Verify bifDB file exists at: "
                  f"{os.path.join(DEFAULT_DB_DIR, 'bifDB_dir', 'bifDB')}")
            sys.exit(1)
        args.bifdb, args.mapping_file, detected_tp_mapping = get_default_paths()
        if detected_tp_mapping and not args.transporter_mapping:
            args.transporter_mapping = detected_tp_mapping

    # Final validation
    if not os.path.exists(args.bifdb):
        print(f"ERROR: Database file not found: {args.bifdb}")
        sys.exit(1)

    if not os.path.exists(args.mapping_file):
        print(f"ERROR: Mapping file not found: {args.mapping_file}")
        sys.exit(1)

    return args


# =============================================================================
# MMSEQS2 SEARCH
# =============================================================================

def run_mmseqs_search(query_file, ref_db, output_prefix, threads, sensitivity):
    """Run MMseqs2 search against reference database."""
    log_message(f"Running MMseqs2 search for {query_file}...")

    tmp_dir = f"{output_prefix}_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        'mmseqs', 'easy-search',
        query_file, ref_db,
        f"{output_prefix}_results.tsv",
        tmp_dir,
        '--format-output',
        'query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits',
        '--threads', str(threads),
        '-s', str(sensitivity)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log_message(f"MMseqs2 search completed for {query_file}")
    except subprocess.CalledProcessError as e:
        error_msg = (f"ERROR: MMseqs2 search failed for {query_file}\n"
                     f"Command: {' '.join(cmd)}\n"
                     f"Error: {e.stderr.decode()}")
        log_message(error_msg)
        return False

    subprocess.run(['rm', '-rf', tmp_dir], check=True)
    return True


# =============================================================================
# GH ANNOTATION
# =============================================================================

def load_mapping_file(mapping_file):
    """Load and process the GH mapping file."""
    log_message(f"Loading mapping file: {mapping_file}")
    try:
        mapping_df = pd.read_csv(mapping_file, sep='\t')
        log_message(f"Loaded {len(mapping_df)} reference sequences")

        required_cols = ['GH_family-F1_threshold', 'GH-cluster threshold']
        missing_cols = [col for col in required_cols if col not in mapping_df.columns]
        if missing_cols:
            log_message(f"ERROR: Missing required threshold columns: {missing_cols}")
            sys.exit(1)

        for col in required_cols:
            mapping_df[col] = pd.to_numeric(mapping_df[col], errors='coerce')
            n_missing = mapping_df[col].isna().sum()
            if n_missing > 0:
                log_message(f"WARNING: {n_missing} rows have missing/invalid values in {col}")

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
    """Process MMseqs2 results and apply annotation thresholds."""
    if not os.path.exists(results_file):
        log_message(f"WARNING: Results file not found: {results_file}")
        return pd.DataFrame()

    try:
        results_df = pd.read_csv(
            results_file, sep='\t',
            names=['query', 'target', 'pident', 'alnlen', 'mismatch',
                   'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits'])
    except Exception as e:
        log_message(f"WARNING: Failed to load results file {results_file}: {e}")
        return pd.DataFrame()

    if len(results_df) == 0:
        log_message(f"WARNING: No hits found for {genome_name}")
        return pd.DataFrame()

    log_message(f"  Raw hits found: {len(results_df)}", print_also=False)

    results_df = results_df.loc[results_df.groupby('query')['bits'].idxmax()]
    log_message(f"  Unique query sequences: {len(results_df)}", print_also=False)

    results_df = results_df.merge(mapping_df, left_on='target',
                                  right_on='Protein_Name', how='left')

    results_df['query_coverage'] = (
        (results_df['qend'] - results_df['qstart'] + 1) / results_df['alnlen'])
    results_df['target_coverage'] = (
        (results_df['tend'] - results_df['tstart'] + 1) / results_df['alnlen'])
    results_df['min_coverage'] = np.minimum(
        results_df['query_coverage'], results_df['target_coverage'])

    annotations = []
    failed_coverage = failed_bitscore = failed_family_threshold = missing_threshold_data = 0

    for _, row in results_df.iterrows():
        if pd.isna(row.get('GH_family-F1_threshold')) or pd.isna(row.get('GH-cluster threshold')):
            missing_threshold_data += 1
            continue

        gh_family_threshold = float(row['GH_family-F1_threshold'])
        cluster_threshold = float(row['GH-cluster threshold'])

        if row['min_coverage'] < COVERAGE_THRESHOLD:
            failed_coverage += 1
            continue
        if row['bits'] < BITSCORE_THRESHOLD:
            failed_bitscore += 1
            continue

        if row['pident'] >= gh_family_threshold:
            annotation = row.copy()
            annotation['Genome'] = genome_name
            if row['pident'] >= cluster_threshold:
                annotation['Assigned_cluster'] = row['Cluster_annotation']
            else:
                annotation['Assigned_cluster'] = f"{row['GH_family']}_cluster_undefined"
            annotations.append(annotation)
        else:
            failed_family_threshold += 1

    if annotations:
        filtered_df = pd.DataFrame(annotations)
        log_message(f"Processed {len(filtered_df)} annotations for {genome_name}")
        n_family_only = len(
            filtered_df[filtered_df['Assigned_cluster'].str.contains('_cluster_undefined')])
        n_cluster = len(
            filtered_df[~filtered_df['Assigned_cluster'].str.contains('_cluster_undefined')])
        log_message(f"  - {n_family_only} sequences: GH family assignment only")
        log_message(f"  - {n_cluster} sequences: GH family + cluster assignment")
        log_message(f"  - Failed coverage filter: {failed_coverage}", print_also=False)
        log_message(f"  - Failed bitscore filter: {failed_bitscore}", print_also=False)
        log_message(f"  - Failed family threshold: {failed_family_threshold}", print_also=False)
        log_message(f"  - Missing threshold data: {missing_threshold_data}", print_also=False)
        log_message(f"  - Mean percent identity: {filtered_df['pident'].mean():.1f}%",
                    print_also=False)
        log_message(f"  - Mean coverage: {filtered_df['min_coverage'].mean():.2f}",
                    print_also=False)
        return filtered_df
    else:
        log_message(f"No annotations passed thresholds for {genome_name}")
        log_message(f"  - Failed coverage filter: {failed_coverage}", print_also=False)
        log_message(f"  - Failed bitscore filter: {failed_bitscore}", print_also=False)
        log_message(f"  - Failed family threshold: {failed_family_threshold}", print_also=False)
        log_message(f"  - Missing threshold data: {missing_threshold_data}", print_also=False)
        return pd.DataFrame()


# =============================================================================
# TRANSPORTER MODULE
# =============================================================================

def load_transporter_mapping(mapping_file):
    """Load and validate the transporter mapping file."""
    log_message(f"Loading transporter mapping file: {mapping_file}")
    try:
        mapping_df = pd.read_csv(mapping_file, sep='\t')
        log_message(f"Loaded {len(mapping_df)} transporter reference entries")

        required_cols = ['Transporter_gene', 'Reference_gene',
                         'Required by bifidoAnnotator clusters']
        missing_cols = [col for col in required_cols if col not in mapping_df.columns]
        if missing_cols:
            log_message(f"ERROR: Missing required columns in transporter mapping: {missing_cols}")
            sys.exit(1)

        log_section("TRANSPORTER MAPPING FILE STATISTICS")
        log_message(f"Total transporter entries: {len(mapping_df)}", print_also=False)
        log_message(f"Unique transporter genes: {mapping_df['Transporter_gene'].nunique()}",
                    print_also=False)
        if 'Transporter_substrate_category' in mapping_df.columns:
            cat_counts = mapping_df['Transporter_substrate_category'].value_counts()
            log_message("Substrate categories:", print_also=False)
            for cat, count in cat_counts.items():
                log_message(f"  {cat}: {count}", print_also=False)

        log_message("Transporter mapping file loaded successfully")
        return mapping_df

    except Exception as e:
        log_message(f"ERROR: Failed to load transporter mapping file: {e}")
        sys.exit(1)


def process_transporter_results(results_file, transporter_mapping_df, genome_name):
    """Process MMseqs2 results for transporters."""
    if not os.path.exists(results_file):
        log_message(f"WARNING: Transporter results file not found: {results_file}")
        return pd.DataFrame()

    try:
        results_df = pd.read_csv(
            results_file, sep='\t',
            names=['query', 'target', 'pident', 'alnlen', 'mismatch',
                   'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits'])
    except Exception as e:
        log_message(f"WARNING: Failed to load transporter results file {results_file}: {e}")
        return pd.DataFrame()

    if len(results_df) == 0:
        log_message(f"  No transporter hits found for {genome_name}")
        return pd.DataFrame()

    log_message(f"  Transporter raw hits: {len(results_df)}", print_also=False)

    results_df = results_df.loc[results_df.groupby('query')['bits'].idxmax()]
    log_message(f"  Unique transporter query sequences: {len(results_df)}", print_also=False)

    # Extract gene name from target: GeneName__Accession format
    results_df['Transporter_gene_match'] = results_df['target'].str.split('__').str[0]

    results_df = results_df.merge(
        transporter_mapping_df,
        left_on='Transporter_gene_match', right_on='Transporter_gene', how='left')
    results_df.drop(columns=['Transporter_gene_match'], inplace=True)

    results_df['query_coverage'] = (
        (results_df['qend'] - results_df['qstart'] + 1) / results_df['alnlen'])
    results_df['target_coverage'] = (
        (results_df['tend'] - results_df['tstart'] + 1) / results_df['alnlen'])
    results_df['min_coverage'] = np.minimum(
        results_df['query_coverage'], results_df['target_coverage'])

    failed_no_mapping = results_df['Transporter_gene'].isna().sum()
    results_df = results_df[results_df['Transporter_gene'].notna()].copy()

    if len(results_df) == 0:
        log_message(f"  No transporter hits matched mapping file for {genome_name}")
        log_message(f"  - No mapping match: {failed_no_mapping}", print_also=False)
        return pd.DataFrame()

    failed_coverage = (results_df['min_coverage'] < COVERAGE_THRESHOLD).sum()
    results_df = results_df[results_df['min_coverage'] >= COVERAGE_THRESHOLD].copy()

    failed_bitscore = (results_df['bits'] < BITSCORE_THRESHOLD).sum()
    results_df = results_df[results_df['bits'] >= BITSCORE_THRESHOLD].copy()

    failed_identity = (results_df['pident'] < TRANSPORTER_IDENTITY_THRESHOLD).sum()
    results_df = results_df[results_df['pident'] >= TRANSPORTER_IDENTITY_THRESHOLD].copy()

    log_message(f"  - No mapping match: {failed_no_mapping}", print_also=False)
    log_message(f"  - Failed coverage filter: {failed_coverage}", print_also=False)
    log_message(f"  - Failed bitscore filter: {failed_bitscore}", print_also=False)
    log_message(f"  - Failed identity filter (<{TRANSPORTER_IDENTITY_THRESHOLD}%): "
                f"{failed_identity}", print_also=False)

    if len(results_df) == 0:
        log_message(f"  No transporter annotations passed thresholds for {genome_name}")
        return pd.DataFrame()

    results_df['Genome'] = genome_name
    log_message(f"  Transporter annotations passing filters: {len(results_df)}")
    return results_df


def build_cluster_to_transporter_map(transporter_mapping_df):
    """Build {cluster_name: [transporter_gene, ...]} from mapping file."""
    cluster_to_transporters = {}
    for _, row in transporter_mapping_df.iterrows():
        transporter_gene = row['Transporter_gene']
        required_str = str(row.get('Required by bifidoAnnotator clusters', ''))
        if not required_str or required_str == 'nan':
            continue
        clusters = [c.strip() for c in required_str.split(';') if c.strip()]
        for cluster in clusters:
            if cluster not in cluster_to_transporters:
                cluster_to_transporters[cluster] = []
            if transporter_gene not in cluster_to_transporters[cluster]:
                cluster_to_transporters[cluster].append(transporter_gene)
    return cluster_to_transporters


def build_genome_transporter_lookup(all_transporter_results):
    """Build {genome_name: set(transporter_gene, ...)}."""
    lookup = {}
    if len(all_transporter_results) == 0:
        return lookup
    for _, row in all_transporter_results.iterrows():
        genome = row['Genome']
        gene = row['Transporter_gene']
        if genome not in lookup:
            lookup[genome] = set()
        lookup[genome].add(gene)
    return lookup


def add_transporter_columns(df, genome_transporter_lookup, cluster_to_transporter_map):
    """Add Transporter_present and Transporter_ID columns to a GH annotation DataFrame."""
    transporter_present_list = []
    transporter_id_list = []

    for _, row in df.iterrows():
        genome = row['Genome']
        assigned_cluster = str(row.get('Assigned_cluster', ''))
        genome_transporters = genome_transporter_lookup.get(genome, set())
        associated_transporters = set()

        if '_cluster_undefined' in assigned_cluster:
            gh_prefix = assigned_cluster.replace('_cluster_undefined', '').strip()
            for cluster_key, t_genes in cluster_to_transporter_map.items():
                if (cluster_key.startswith(gh_prefix + ' - ') or
                        cluster_key.startswith(gh_prefix + '-')):
                    associated_transporters.update(t_genes)
        else:
            associated_transporters = set(
                cluster_to_transporter_map.get(assigned_cluster, []))

        found_transporters = associated_transporters & genome_transporters

        if found_transporters:
            transporter_present_list.append('Yes')
            transporter_id_list.append('; '.join(sorted(found_transporters)))
        else:
            transporter_present_list.append('No')
            transporter_id_list.append('')

    df = df.copy()
    df['Transporter_present'] = transporter_present_list
    df['Transporter_ID'] = transporter_id_list
    return df


def generate_transporter_annotations(all_transporter_results, output_dir):
    """Generate detailed transporter annotation table."""
    if len(all_transporter_results) == 0:
        log_message("WARNING: No transporter results to write")
        return

    base_cols = ['query', 'Genome', 'Transporter_substrate_category', 'Transporter_gene',
                 'Transporter description', 'Reference_gene', 'Reference_species',
                 'Required by bifidoAnnotator clusters', 'pident', 'bits', 'evalue']
    available_cols = [c for c in base_cols if c in all_transporter_results.columns]

    out_df = all_transporter_results[available_cols].copy()
    out_path = os.path.join(output_dir, 'bifidoAnnotator_tables',
                            'transporter_detailed_annotations.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    log_message("Generated transporter detailed annotations table")


def generate_transporter_genome_summary(all_transporter_results, output_dir, all_genome_names):
    """Generate per-genome transporter summary with copy numbers."""
    if len(all_transporter_results) == 0:
        return

    groupby_cols = [c for c in ['Genome', 'Transporter_gene']
                    if c in all_transporter_results.columns]
    agg_cols = ['Transporter_substrate_category', 'Transporter description',
                'Reference_gene', 'Reference_species',
                'Required by bifidoAnnotator clusters']
    agg_dict = {c: 'first' for c in agg_cols if c in all_transporter_results.columns}
    agg_dict['query'] = 'count'

    summary = all_transporter_results.groupby(groupby_cols).agg(agg_dict).reset_index()
    summary.rename(columns={'query': 'copy_number'}, inplace=True)

    if all_genome_names:
        genomes_with_annotations = set(all_transporter_results['Genome'].unique())
        missing_genomes = set(all_genome_names) - genomes_with_annotations
        if missing_genomes:
            log_message(
                f"Adding {len(missing_genomes)} genomes with no transporter annotations to summary")
            missing_rows = []
            for genome in missing_genomes:
                row = {'Genome': genome, 'Transporter_gene': pd.NA, 'copy_number': pd.NA}
                for col in agg_dict:
                    if col != 'query':
                        row[col] = pd.NA
                missing_rows.append(row)
            summary = pd.concat([summary, pd.DataFrame(missing_rows)], ignore_index=True)

    out_path = os.path.join(output_dir, 'bifidoAnnotator_tables',
                            'transporter_genome_summary.tsv')
    summary.to_csv(out_path, sep='\t', index=False)
    log_message("Generated transporter genome summary table")


# =============================================================================
# OUTPUT TABLES
# =============================================================================

def create_output_structure(output_dir):
    """Create output directory structure."""
    subdirs = ['bifidoAnnotator_tables', 'bifidoAnnotator_visualizations']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


def generate_detailed_annotations(all_results, output_dir,
                                   genome_transporter_lookup=None,
                                   cluster_to_transporter_map=None):
    """Generate detailed annotation tables."""
    if len(all_results) == 0:
        log_message("WARNING: No results to process")
        return

    base_columns = ['query', 'Genome', 'GH_family', 'Enzyme', 'Cluster_ID',
                    'Assigned_cluster', 'Validation_status', 'Reference',
                    'GH_localization', 'HMG-utilization', 'pident', 'bits', 'evalue']
    available_columns = list(all_results.columns)
    final_columns = [col for col in base_columns if col in available_columns]
    detailed_df = all_results[final_columns].copy()

    if genome_transporter_lookup is not None and cluster_to_transporter_map is not None:
        log_message("Adding transporter linkage columns to detailed annotations...")
        detailed_df = add_transporter_columns(
            detailed_df, genome_transporter_lookup, cluster_to_transporter_map)

    detailed_df.to_csv(
        os.path.join(output_dir, 'bifidoAnnotator_tables', 'detailed_annotations.tsv'),
        sep='\t', index=False)
    log_message("Generated detailed annotations table")


def generate_genome_summary(all_results, output_dir, all_genome_names,
                             genome_transporter_lookup=None,
                             cluster_to_transporter_map=None):
    """Generate per-genome summary with copy numbers."""
    if len(all_results) == 0:
        return

    available_columns = list(all_results.columns)
    exclude_from_agg = [
        'query', 'target', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend',
        'tstart', 'tend', 'pident', 'bits', 'evalue', 'query_coverage',
        'target_coverage', 'min_coverage', 'GH_family-F1_threshold',
        'GH-cluster threshold', 'Protein_Name'
    ]

    agg_dict = {'query': 'count'}
    additional_agg_columns = []
    for col in available_columns:
        if (col not in exclude_from_agg and
                col not in ['Genome', 'GH_family', 'Enzyme', 'Assigned_cluster']):
            agg_dict[col] = 'first'
            additional_agg_columns.append(col)

    genome_summary = all_results.groupby(
        ['Genome', 'GH_family', 'Enzyme', 'Assigned_cluster']).agg(agg_dict).reset_index()
    genome_summary.rename(columns={'query': 'copy_number'}, inplace=True)

    if all_genome_names:
        genomes_with_annotations = set(all_results['Genome'].unique())
        missing_genomes = set(all_genome_names) - genomes_with_annotations
        if missing_genomes:
            log_message(
                f"Adding {len(missing_genomes)} genomes with no annotations to genome summary")
            missing_rows = []
            for genome in missing_genomes:
                missing_row = {
                    'Genome': genome, 'GH_family': pd.NA,
                    'Enzyme': pd.NA, 'Assigned_cluster': pd.NA, 'copy_number': pd.NA
                }
                for col in additional_agg_columns:
                    missing_row[col] = pd.NA
                missing_rows.append(missing_row)
            genome_summary = pd.concat(
                [genome_summary, pd.DataFrame(missing_rows)], ignore_index=True)

    if genome_transporter_lookup is not None and cluster_to_transporter_map is not None:
        log_message("Adding transporter linkage columns to genome summary...")
        has_cluster = genome_summary['Assigned_cluster'].notna()
        if has_cluster.any():
            annotated_rows = add_transporter_columns(
                genome_summary[has_cluster].copy(),
                genome_transporter_lookup, cluster_to_transporter_map)
            genome_summary.loc[has_cluster, 'Transporter_present'] = (
                annotated_rows['Transporter_present'].values)
            genome_summary.loc[has_cluster, 'Transporter_ID'] = (
                annotated_rows['Transporter_ID'].values)
        genome_summary['Transporter_present'] = (
            genome_summary['Transporter_present'].fillna(''))
        genome_summary['Transporter_ID'] = genome_summary['Transporter_ID'].fillna('')

    genome_summary.to_csv(
        os.path.join(output_dir, 'bifidoAnnotator_tables', 'genome_summary.tsv'),
        sep='\t', index=False)
    log_message("Generated genome summary table")


def generate_wide_matrices(all_results, output_dir, all_genome_names):
    """Generate wide-format matrices for GH family, enzyme, and cluster."""
    if len(all_results) == 0:
        log_message("WARNING: No results to process")
        return None, None, None

    gh_matrix = all_results.groupby(['Genome', 'GH_family']).size().unstack(fill_value=0)
    enzyme_matrix = all_results.groupby(['Genome', 'Enzyme']).size().unstack(fill_value=0)
    cluster_matrix = all_results.groupby(
        ['Genome', 'Assigned_cluster']).size().unstack(fill_value=0)

    if all_genome_names:
        genomes_with_annotations = set(all_results['Genome'].unique())
        missing_genomes = set(all_genome_names) - genomes_with_annotations
        if missing_genomes:
            log_message(
                f"Adding {len(missing_genomes)} genomes with no annotations to matrices")
            for genome in missing_genomes:
                for col in gh_matrix.columns:
                    gh_matrix.loc[genome, col] = 0
                for col in enzyme_matrix.columns:
                    enzyme_matrix.loc[genome, col] = 0
                for col in cluster_matrix.columns:
                    cluster_matrix.loc[genome, col] = 0

    gh_matrix.to_csv(
        os.path.join(output_dir, 'bifidoAnnotator_tables', 'gh_family_matrix.tsv'), sep='\t')
    enzyme_matrix.to_csv(
        os.path.join(output_dir, 'bifidoAnnotator_tables', 'enzyme_matrix.tsv'), sep='\t')
    cluster_matrix.to_csv(
        os.path.join(output_dir, 'bifidoAnnotator_tables', 'cluster_matrix.tsv'), sep='\t')

    log_message("Generated wide-format matrices")
    return gh_matrix, enzyme_matrix, cluster_matrix


# =============================================================================
# HEATMAP GENERATOR
# =============================================================================

class HeatmapGenerator:
    """Advanced heatmap generator with improved positioning and color handling."""

    def __init__(self, output_dir, annotations_file=None, heatmap_col='blue',
                 hmg_filter_mode='hmg_unknown', cluster_hmg_map=None):
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / 'bifidoAnnotator_tables'
        self.vis_dir = self.output_dir / 'bifidoAnnotator_visualizations'
        self.annotations_file = annotations_file
        self.heatmap_col = heatmap_col
        self.hmg_filter_mode = hmg_filter_mode
        self.cluster_hmg_map = cluster_hmg_map if cluster_hmg_map is not None else {}

        self.gh_matrix_file = self.tables_dir / 'gh_family_matrix.tsv'
        self.cluster_matrix_file = self.tables_dir / 'cluster_matrix.tsv'
        self.enzyme_matrix_file = self.tables_dir / 'enzyme_matrix.tsv'

        self.gh_matrix = None
        self.cluster_matrix = None
        self.enzyme_matrix = None
        self.annotations = None
        self.annotation_columns = []
        self.annotation_colors = {}

        self.scale_factor = 1.0
        self.font_scale = 1.0
        self.global_vmin = 0
        self.global_vmax = 1
        self.global_cmap = None
        self.global_discrete_levels = [0, 1]

    def calculate_adaptive_figsize(self, n_genomes, n_features, has_annotations=False):
        if n_genomes <= 20:
            base_width = 8 + (n_genomes * 0.15)
        elif n_genomes <= 50:
            base_width = 11 + ((n_genomes - 20) * 0.2)
        elif n_genomes <= 150:
            base_width = 17 + ((n_genomes - 50) * 0.08)
        else:
            base_width = 25 + ((n_genomes - 150) * 0.03)

        if n_features <= 10:
            base_height = 6 + (n_features * 0.3)
        elif n_features <= 50:
            base_height = 9 + ((n_features - 10) * 0.22)
        elif n_features <= 150:
            base_height = 18 + ((n_features - 50) * 0.10)
        else:
            base_height = 28 + ((n_features - 150) * 0.04)

        if has_annotations and hasattr(self, 'annotation_columns'):
            n_annotation_cols = len(self.annotation_columns)
            base_height += (n_annotation_cols * 0.8)
            base_width += 2

        width = max(8, min(35, base_width))
        height = max(6, min(35, base_height))
        return (int(width), int(height))

    def determine_figsize(self, matrix_name, manual_figsize=None):
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
            log_message(
                f"  Cluster heatmap: increased height to {calculated_size[1]} inches",
                print_also=False)

        size_type = "with annotations" if has_annotations else "basic"
        log_message(
            f"  Auto-calculated size {calculated_size[0]}×{calculated_size[1]} inches "
            f"({n_genomes} genomes, {n_features} features, {size_type})",
            print_also=False)

        return calculated_size

    def load_data(self):
        log_message("Loading matrix files for visualization...")

        if not all([self.gh_matrix_file.exists(), self.cluster_matrix_file.exists(),
                    self.enzyme_matrix_file.exists()]):
            log_message("ERROR: Matrix files not found.")
            return False

        try:
            self.gh_matrix = pd.read_csv(self.gh_matrix_file, sep='\t', index_col=0)
        except Exception as e:
            log_message(f"Error loading GH matrix: {e}")
            return False

        try:
            self.cluster_matrix = pd.read_csv(self.cluster_matrix_file, sep='\t', index_col=0)
        except Exception as e:
            log_message(f"Error loading cluster matrix: {e}")
            return False

        try:
            self.enzyme_matrix = pd.read_csv(self.enzyme_matrix_file, sep='\t', index_col=0)
        except Exception as e:
            log_message(f"Error loading enzyme matrix: {e}")
            return False

        if self.annotations_file:
            try:
                self.annotations = pd.read_csv(self.annotations_file, sep='\t', index_col=0)
                self.annotation_columns = list(self.annotations.columns)
                log_message(
                    f"  Annotations: {self.annotations.shape}, "
                    f"columns: {self.annotation_columns}", print_also=False)
                for col in self.annotation_columns:
                    if 'species' in col.lower():
                        self.annotations[col] = (self.annotations[col].astype(str)
                                                  .str.replace('s__', '', regex=False))
            except Exception as e:
                log_message(f"Warning: Failed to load annotations file: {e}")
                self.annotations = None
                self.annotation_columns = []
        else:
            log_message("  No annotations file provided - generating basic heatmaps")
            self.annotations = None
            self.annotation_columns = []

        return self.validate_data()

    def validate_data(self):
        gh_genomes = set(self.gh_matrix.index)
        cluster_genomes = set(self.cluster_matrix.index)
        enzyme_genomes = set(self.enzyme_matrix.index)
        common_genomes = gh_genomes & cluster_genomes & enzyme_genomes

        if self.annotations is not None:
            annotation_genomes = set(self.annotations.index)
            common_genomes = common_genomes & annotation_genomes

        if len(common_genomes) == 0:
            log_message("ERROR: No common genomes found!")
            return False

        common_genomes_sorted = sorted(common_genomes)
        self.gh_matrix = self.gh_matrix.loc[common_genomes_sorted]
        self.cluster_matrix = self.cluster_matrix.loc[common_genomes_sorted]
        self.enzyme_matrix = self.enzyme_matrix.loc[common_genomes_sorted]

        if self.annotations is not None:
            self.annotations = self.annotations.loc[common_genomes_sorted]

        log_message(f"Using {len(common_genomes)} common genomes for visualization",
                    print_also=False)
        return True

    def calculate_color_scale_for_data(self, data_matrix):
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
                elif vmax == 2:
                    colors.extend([blue_palette[2], blue_palette[6]])
                elif vmax == 3:
                    colors.extend([blue_palette[1], blue_palette[4], blue_palette[7]])
                elif vmax == 4:
                    colors.extend([blue_palette[1], blue_palette[3],
                                   blue_palette[5], blue_palette[8]])
                elif vmax == 5:
                    colors.extend([blue_palette[0], blue_palette[2], blue_palette[4],
                                   blue_palette[6], blue_palette[8]])
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
                    colors.extend([red_palette[1], red_palette[3],
                                   red_palette[5], red_palette[7]])
                elif vmax == 5:
                    colors.extend([red_palette[0], red_palette[2], red_palette[4],
                                   red_palette[6], red_palette[8]])
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
            return ('Reds' if self.heatmap_col == 'red' else 'Blues'), 0, 1, [0, 1]

    def setup_annotation_colors(self):
        if self.annotations is None or len(self.annotation_columns) == 0:
            return

        dark2_palette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a',
                         '#66a61e', '#e6ab02', '#a6761d', '#666666']
        set3_colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                       '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']

        color_palettes = [
            dark2_palette,
            ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
             '#ff7f00', '#ffff33', '#a65628', '#f781bf'],
            ['#404040', '#595959', '#737373', '#8c8c8c', '#a6a6a6',
             '#bfbfbf', '#d9d9d9', '#e6e6e6', '#f0f0f0'],
            set3_colors
        ]

        n_cols = len(self.annotation_columns)

        for i, col in enumerate(self.annotation_columns):
            self.annotations[col] = self.annotations[col].fillna('N.A.')
            unique_values = sorted(self.annotations[col].unique())
            n_values = len(unique_values)
            n_non_na_values = len([v for v in unique_values if v != 'N.A.'])

            if i == n_cols - 1:
                palette_idx = 3
                chosen_palette = color_palettes[3]
                is_dark2_palette = False
            elif i == n_cols - 2:
                palette_idx = 2
                chosen_palette = color_palettes[2]
                is_dark2_palette = False
            else:
                available_palettes = 2
                palette_idx = i % available_palettes
                chosen_palette = color_palettes[palette_idx]
                is_dark2_palette = (palette_idx == 0)

            color_dict = {}
            palette_size = len(chosen_palette)

            if n_non_na_values <= palette_size:
                if is_dark2_palette:
                    if n_non_na_values == 1:
                        color_indices = [0]
                    elif n_non_na_values == 2:
                        color_indices = [0, 1]
                    elif n_non_na_values == 3:
                        color_indices = [0, 1, 2]
                    elif n_non_na_values == 4:
                        color_indices = [0, 1, 3, 4]
                    elif n_non_na_values == 5:
                        color_indices = [0, 1, 2, 3, 4]
                    elif n_non_na_values == 6:
                        color_indices = [0, 1, 2, 3, 4, 5]
                    elif n_non_na_values == 7:
                        color_indices = [0, 1, 2, 3, 4, 5, 6]
                    else:
                        color_indices = list(range(8))
                else:
                    if palette_idx == 2:  # Grayscale
                        if n_non_na_values == 1:
                            color_indices = [4]
                        elif n_non_na_values == 2:
                            color_indices = [1, 8]
                        elif n_non_na_values == 3:
                            color_indices = [0, 4, 8]
                        elif n_non_na_values == 4:
                            color_indices = [0, 3, 6, 8]
                        elif n_non_na_values == 5:
                            color_indices = [0, 2, 4, 6, 8]
                        else:
                            step = (palette_size - 1) / (n_non_na_values - 1)
                            color_indices = [int(round(j * step))
                                             for j in range(n_non_na_values)]
                    else:
                        color_indices = list(range(n_non_na_values))
            else:
                color_indices = list(range(n_non_na_values))

            color_idx = 0
            for value in unique_values:
                if value == 'N.A.':
                    color_dict[value] = '#F0F0F0'
                else:
                    if n_non_na_values > 0:
                        palette_idx_to_use = color_indices[color_idx] % palette_size
                        color_dict[value] = chosen_palette[palette_idx_to_use]
                        color_idx += 1
                    else:
                        color_dict[value] = '#CCCCCC'

            self.annotation_colors[col] = color_dict

    def setup_hmg_colors(self):
        """Define row annotation colors for HMG-utilization values."""
        yes_color = '#990000' if self.heatmap_col == 'blue' else '#005C99'
        return {
            'Yes':     yes_color,
            'No':      '#1a1a1a',
            'Unknown': '#F5C800',
            'N.A.':    '#d9d9d9',
        }

    def add_hmg_row_legend(self, fig, hmg_colors, figsize, position=None,
                           present_values=None):
        """Add a compact legend for the HMG-utilization row annotation bar."""
        legend_font_size = max(8, min(11, figsize[1] * 0.4))
        patches = [mpatches.Patch(color=color, label=label,
                                  edgecolor='black', linewidth=0.8)
                   for label, color in hmg_colors.items()
                   if label != 'N.A.' and (present_values is None or label in present_values)]

        legend_left, legend_bottom = position if position is not None else (0.01, 0.05)

        legend = fig.legend(handles=patches,
                            title='HMG-utilization',
                            bbox_to_anchor=(legend_left, legend_bottom),
                            bbox_transform=fig.transFigure,
                            loc='upper left',
                            ncol=1,
                            fontsize=legend_font_size,
                            title_fontsize=legend_font_size,
                            frameon=False,
                            columnspacing=0.4,
                            handletextpad=0.2,
                            handlelength=0.8,
                            borderaxespad=0,
                            prop={'family': 'Nimbus Sans', 'size': legend_font_size})
        legend.get_title().set_fontfamily('Nimbus Sans')
        legend.get_title().set_fontweight('bold')

    def calculate_dynamic_positions(self, heatmap_ax, fig, n_annotation_rows,
                                     show_column_labels, n_features):
        """Calculate positions using tight bbox to avoid overlaps."""
        fig.canvas.draw()
        hm_pos = heatmap_ax.get_position()
        hm_left = hm_pos.x0
        hm_bottom = hm_pos.y0
        hm_width = hm_pos.width
        hm_height = hm_pos.height
        hm_right = hm_left + hm_width

        try:
            renderer = fig.canvas.get_renderer()
            tight_bbox = heatmap_ax.get_tightbbox(renderer)
            fig_bbox = fig.transFigure.inverted().transform(
                [[tight_bbox.x0, tight_bbox.y0], [tight_bbox.x1, tight_bbox.y1]])
            content_bottom = fig_bbox[0, 1]
        except Exception:
            content_bottom = hm_bottom - (0.10 if show_column_labels else 0.04)

        cbar_width = 0.12
        cbar_height = 0.025
        cbar_left = hm_right + 0.02
        cbar_bottom = hm_bottom - 0.05

        legend_gap = 0.015
        legend_bottom = content_bottom - legend_gap
        legend_left = hm_left + (hm_width * 0.1)

        return {
            'colorbar': (cbar_left, cbar_bottom, cbar_width, cbar_height),
            'legend': (legend_left, legend_bottom),
            'heatmap': (hm_left, hm_bottom, hm_width, hm_height),
        }

    def create_clustermap(self, data, title, output_file_base,
                          manual_figsize=None, matrix_name='unknown'):
        """Create a hierarchically clustered heatmap."""
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

        # HMG row filtering and annotation bar
        row_colors_series = None
        hmg_colors = None
        if matrix_name == 'cluster' and self.cluster_hmg_map:
            hmg_colors = self.setup_hmg_colors()

            if self.hmg_filter_mode in ('hmg_unknown', 'hmg_only'):
                keep_values = (('Yes',) if self.hmg_filter_mode == 'hmg_only'
                               else ('Yes', 'Unknown'))
                hmg_keep = []
                for cluster in data_filtered.index:
                    val = self.cluster_hmg_map.get(cluster, 'Unknown')
                    if not val or pd.isna(val):
                        val = 'Unknown'
                    hmg_keep.append(val in keep_values)
                data_filtered = data_filtered[hmg_keep]
                log_message(
                    f"  HMG filter ({'/'.join(keep_values)} only): "
                    f"{sum(hmg_keep)} of {len(hmg_keep)} clusters retained",
                    print_also=False)

                if len(data_filtered) == 0:
                    log_message("WARNING: No clusters remaining after HMG filter")
                    return None

            if self.hmg_filter_mode != 'hmg_only':
                row_colors_series = True
            else:
                log_message(
                    "  HMG row annotation bar suppressed (--hmg-only: all rows are Yes)",
                    print_also=False)

        n_genomes = len(data_filtered.columns)
        show_column_labels = n_genomes <= 150
        log_message(f"  Genomes: {n_genomes}, Show labels: {show_column_labels}",
                    print_also=False)

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
                colors_ratio = max(0.035, (colors_ratio / 1.3) * 0.85)
            else:
                colors_ratio = max(0.03, min(0.20, colors_ratio))
        else:
            colors_ratio = 0

        clustermap_kwargs = dict(
            method='average',
            metric='euclidean',
            cmap=cmap,
            norm=norm,
            figsize=figsize,
            dendrogram_ratio=dendrogram_ratio,
            linewidths=1.2,
            linecolor='black',
            xticklabels=show_column_labels,
            yticklabels=True,
            tree_kws={'linewidths': 1.5}
        )
        if annotation_color_lists:
            clustermap_kwargs['col_colors'] = annotation_color_lists
            clustermap_kwargs['colors_ratio'] = colors_ratio

        g = sns.clustermap(data_filtered, **clustermap_kwargs)

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

        plt.subplots_adjust(left=0.12, right=0.75, top=top_space, bottom=bottom_space)
        g.fig.canvas.draw()

        g.ax_heatmap.set_xlabel('Genomes', fontsize=10, fontweight='bold',
                                 fontfamily='Nimbus Sans')
        g.ax_heatmap.set_ylabel('Features', fontsize=10, fontweight='bold',
                                 fontfamily='Nimbus Sans')

        if show_column_labels:
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8,
                     fontfamily='Nimbus Sans')
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8,
                 fontfamily='Nimbus Sans')

        n_annotation_rows = len(annotation_color_lists) if annotation_color_lists else 0
        positions = self.calculate_dynamic_positions(
            g.ax_heatmap, g.fig, n_annotation_rows, show_column_labels, len(data_filtered))

        if annotation_color_lists:
            self.add_annotation_labels(g.fig, g)
            self.create_annotation_separation(g)
            legend_end_x = self.add_multiple_legends(g.fig, positions['legend'], figsize)

        # Draw HMG row annotation bar using blended transform
        if row_colors_series is not None and hmg_colors is not None:
            import matplotlib.transforms as mtransforms

            ax_hm = g.ax_heatmap
            g.fig.canvas.draw()
            hm_pos = ax_hm.get_position()

            bar_width_ax = 0.045
            gap_ax = 0.020
            bar_right_ax = -gap_ax
            bar_left_ax = bar_right_ax - bar_width_ax

            trans = mtransforms.blended_transform_factory(
                ax_hm.transAxes,
                ax_hm.transData
            )

            shift_fig = (bar_width_ax + gap_ax) * hm_pos.width
            if hasattr(g, 'ax_row_dendrogram') and g.ax_row_dendrogram is not None:
                dend_pos = g.ax_row_dendrogram.get_position()
                g.ax_row_dendrogram.set_position([
                    dend_pos.x0 - shift_fig,
                    dend_pos.y0,
                    dend_pos.width,
                    dend_pos.height
                ])

            n_rows = len(data_filtered.index)
            if hasattr(g, 'dendrogram_row') and g.dendrogram_row is not None:
                row_order = g.dendrogram_row.reordered_ind
            else:
                row_order = list(range(n_rows))

            for display_row, orig_idx in enumerate(row_order):
                cluster = data_filtered.index[orig_idx]
                val = self.cluster_hmg_map.get(cluster, '')
                if not val or (isinstance(val, float) and pd.isna(val)):
                    val = 'Unknown'
                color = hmg_colors.get(val, hmg_colors['Unknown'])
                rect = mpatches.Rectangle(
                    (bar_left_ax, display_row),
                    bar_width_ax, 1,
                    transform=trans,
                    facecolor=color, edgecolor='black', linewidth=0.8,
                    clip_on=False, zorder=5
                )
                ax_hm.add_patch(rect)

            bar_center_fig = hm_pos.x0 + (bar_left_ax + bar_width_ax / 2) * hm_pos.width
            label_y_fig = hm_pos.y0 - 0.01
            g.fig.text(
                bar_center_fig, label_y_fig,
                'HMG-utilization',
                fontsize=11, fontfamily='Nimbus Sans', fontweight='bold',
                rotation=90, va='top', ha='center',
                transform=g.fig.transFigure
            )

            log_message(
                f"  Drew HMG row bar ({n_rows} rows, blended transform)",
                print_also=False)

        # HMG legend
        if row_colors_series is not None and hmg_colors is not None:
            present_hmg_values = set(
                self.cluster_hmg_map.get(cluster, 'Unknown') or 'Unknown'
                for cluster in data_filtered.index
            )
            if annotation_color_lists:
                _, legend_bottom = positions['legend']
                hmg_legend_position = (legend_end_x, legend_bottom)
            else:
                cbar_left, cbar_bottom, cbar_width, cbar_height = positions['colorbar']
                hmg_legend_position = (cbar_left + cbar_width + 0.03, cbar_bottom)
            self.add_hmg_row_legend(g.fig, hmg_colors, figsize,
                                    position=hmg_legend_position,
                                    present_values=present_hmg_values)

        self.add_dynamic_colorbar(g.fig, positions['colorbar'], cmap, vmin, vmax,
                                   discrete_levels)

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
                       fontsize=9, fontfamily='Nimbus Sans', ha='center')

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
            label.set_fontfamily('Nimbus Sans')
            label.set_fontsize(8)

        cbar_ax.patch.set_facecolor('white')
        cbar_ax.patch.set_alpha(1.0)
        return cbar

    def add_multiple_legends(self, fig, position, figsize):
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
                                prop={'family': 'Nimbus Sans', 'size': legend_font_size})
            legend.get_title().set_fontfamily('Nimbus Sans')
            legend.get_title().set_fontweight('bold')
            current_x += est_width + spacing

        return current_x

    def create_annotation_separation(self, clustermap_obj):
        separation_gap = 0.006
        heatmap_pos = clustermap_obj.ax_heatmap.get_position()
        if (hasattr(clustermap_obj, 'ax_col_colors') and
                clustermap_obj.ax_col_colors is not None):
            col_colors_pos = clustermap_obj.ax_col_colors.get_position()
            new_heatmap_top = col_colors_pos.y0 - separation_gap
            new_heatmap_bottom = new_heatmap_top - heatmap_pos.height
            clustermap_obj.ax_heatmap.set_position(
                [heatmap_pos.x0, new_heatmap_bottom,
                 heatmap_pos.width, heatmap_pos.height])
            if (hasattr(clustermap_obj, 'ax_row_dendrogram') and
                    clustermap_obj.ax_row_dendrogram is not None):
                row_dend_pos = clustermap_obj.ax_row_dendrogram.get_position()
                clustermap_obj.ax_row_dendrogram.set_position(
                    [row_dend_pos.x0, new_heatmap_bottom,
                     row_dend_pos.width, heatmap_pos.height])
            clustermap_obj.fig.canvas.draw_idle()

    def add_annotation_labels(self, fig, clustermap_obj):
        if (hasattr(clustermap_obj, 'ax_col_colors') and
                clustermap_obj.ax_col_colors is not None):
            col_colors_pos = clustermap_obj.ax_col_colors.get_position()
            n_annotations = len(self.annotation_columns)
            row_height = col_colors_pos.height / n_annotations
            for i, col in enumerate(self.annotation_columns):
                label_x = col_colors_pos.x0 + col_colors_pos.width + 0.01
                label_y = (col_colors_pos.y0 + col_colors_pos.height
                           - (i + 0.5) * row_height)
                fig.text(label_x, label_y, col,
                         rotation=0,
                         verticalalignment='center',
                         horizontalalignment='left',
                         fontsize=11,
                         fontweight='bold',
                         fontfamily='Nimbus Sans')

    def generate_heatmaps(self, gh_figsize=None, cluster_figsize=None, enzyme_figsize=None):
        log_message("Generating publication-ready heatmaps...")

        if not self.load_data():
            log_message("ERROR: Failed to load data for visualization")
            return False

        if self.annotations is not None:
            self.setup_annotation_colors()

        annotation_info = (f" with {len(self.annotation_columns)} annotation columns"
                           if self.annotations is not None else " (basic heatmaps)")
        log_message(f"Creating heatmaps{annotation_info} - each with adaptive color scale")

        try:
            self.create_clustermap(
                self.gh_matrix, 'GH Family Abundance Heatmap',
                str(self.vis_dir / 'gh_family_heatmap'),
                manual_figsize=gh_figsize, matrix_name='gh')
        except Exception as e:
            log_message(f"ERROR: Failed to create GH heatmap: {e}")

        try:
            self.create_clustermap(
                self.cluster_matrix, 'Cluster Abundance Heatmap',
                str(self.vis_dir / 'cluster_heatmap'),
                manual_figsize=cluster_figsize, matrix_name='cluster')
        except Exception as e:
            log_message(f"ERROR: Failed to create cluster heatmap: {e}")

        try:
            self.create_clustermap(
                self.enzyme_matrix, 'Enzyme Abundance Heatmap',
                str(self.vis_dir / 'enzyme_heatmap'),
                manual_figsize=enzyme_figsize, matrix_name='enzyme')
        except Exception as e:
            log_message(f"ERROR: Failed to create enzyme heatmap: {e}")

        log_message(f"Heatmap generation completed! Files saved in: {self.vis_dir}")
        return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    print("=" * 80)
    print("Combined bifidoAnnotator: Complete GH Annotation & Visualization Pipeline")
    print("=" * 80)

    # Parse arguments (triggers auto-download if database missing)
    args = parse_arguments()

    # Check dependencies
    check_dependencies()

    # Validate arguments
    if args.genome_directory and not args.sample_file:
        print("ERROR: --sample_file is required when using --genome_directory")
        sys.exit(1)

    if bool(args.transporter_db) != bool(args.transporter_mapping):
        print("ERROR: --transporter_db and --transporter_mapping must both be provided together")
        sys.exit(1)
    run_transporter_module = bool(args.transporter_db)

    # Create output structure
    create_output_structure(args.output_dir)

    # Initialize logging
    initialize_log(args.output_dir, args)

    # Load GH mapping file
    mapping_df = load_mapping_file(args.mapping_file)

    # Load transporter mapping if module is enabled
    transporter_mapping_df = None
    cluster_to_transporter_map = {}
    if run_transporter_module:
        log_message("\nTransporter module enabled")
        transporter_mapping_df = load_transporter_mapping(args.transporter_mapping)
        cluster_to_transporter_map = build_cluster_to_transporter_map(transporter_mapping_df)
        log_message(
            f"Built cluster-to-transporter map: {len(cluster_to_transporter_map)} cluster entries")

    log_section("INPUT PROCESSING")

    # Determine input files
    input_files = []
    genome_names = []

    if args.input_file:
        input_files.append(args.input_file)
        genome_names.append(Path(args.input_file).stem)
    else:
        with open(args.sample_file, 'r') as f:
            for line in f:
                genome_name = line.strip()
                if genome_name:
                    for ext in ['.fasta', '.fa', '.faa']:
                        fasta_path = os.path.join(
                            args.genome_directory, f"{genome_name}{ext}")
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
    log_section("GENOME PROCESSING")

    all_results = []
    all_transporter_results = []

    for input_file, genome_name in zip(input_files, genome_names):
        log_message(f"\nProcessing: {genome_name}")
        log_message(f"File size: {os.path.getsize(input_file) / 1024:.1f} KB",
                    print_also=False)

        # GH annotation
        output_prefix = os.path.join(args.output_dir, genome_name)
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

        # Transporter annotation (optional)
        if run_transporter_module:
            t_output_prefix = os.path.join(args.output_dir, f"{genome_name}_transporter")
            t_success = run_mmseqs_search(input_file, args.transporter_db, t_output_prefix,
                                          args.threads, args.sensitivity)
            if t_success:
                t_results_file = f"{t_output_prefix}_results.tsv"
                t_genome_results = process_transporter_results(
                    t_results_file, transporter_mapping_df, genome_name)
                if not t_genome_results.empty:
                    all_transporter_results.append(t_genome_results)
                if os.path.exists(t_results_file):
                    os.remove(t_results_file)
            else:
                log_message(f"Failed to run transporter search for {genome_name}")

    log_section("OUTPUT GENERATION")

    # Build transporter lookup
    combined_transporter_results = pd.DataFrame()
    genome_transporter_lookup = {}
    if run_transporter_module and all_transporter_results:
        combined_transporter_results = pd.concat(all_transporter_results, ignore_index=True)
        genome_transporter_lookup = build_genome_transporter_lookup(
            combined_transporter_results)
        log_message(
            f"Transporter annotations: {len(combined_transporter_results)} total across "
            f"{combined_transporter_results['Genome'].nunique()} genome(s)")
    elif run_transporter_module:
        log_message("WARNING: No transporter annotations found across all input genomes")

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)

        log_message("\nGenerating output files...")
        generate_detailed_annotations(
            combined_results, args.output_dir,
            genome_transporter_lookup if run_transporter_module else None,
            cluster_to_transporter_map if run_transporter_module else None)
        generate_genome_summary(
            combined_results, args.output_dir, genome_names,
            genome_transporter_lookup if run_transporter_module else None,
            cluster_to_transporter_map if run_transporter_module else None)
        matrices = generate_wide_matrices(combined_results, args.output_dir, genome_names)

        if run_transporter_module:
            log_section("TRANSPORTER OUTPUT GENERATION")
            generate_transporter_annotations(combined_transporter_results, args.output_dir)
            generate_transporter_genome_summary(
                combined_transporter_results, args.output_dir, genome_names)

        # Build cluster → HMG-utilization map
        cluster_hmg_map = {}
        if 'HMG-utilization' in combined_results.columns:
            cluster_hmg_map = (
                combined_results[
                    combined_results['HMG-utilization'].notna() &
                    (combined_results['HMG-utilization'] != '')]
                .drop_duplicates(subset='Assigned_cluster')
                .set_index('Assigned_cluster')['HMG-utilization']
                .to_dict()
            )
            log_message(
                f"Built cluster HMG-utilization map: {len(cluster_hmg_map)} entries",
                print_also=False)

        if args.hmg_only:
            hmg_filter_mode = 'hmg_only'
        elif args.all_genes:
            hmg_filter_mode = 'all'
        else:
            hmg_filter_mode = 'hmg_unknown'

        log_section("VISUALIZATION GENERATION")
        heatmap_generator = HeatmapGenerator(
            args.output_dir, args.annotations_file, args.heatmap_col,
            hmg_filter_mode=hmg_filter_mode, cluster_hmg_map=cluster_hmg_map)
        heatmap_success = heatmap_generator.generate_heatmaps(
            gh_figsize=args.gh_figsize,
            cluster_figsize=args.cluster_figsize,
            enzyme_figsize=args.enzyme_figsize)

        if heatmap_success:
            log_message("Advanced heatmap generation completed successfully")
        else:
            log_message("WARNING: Heatmap generation encountered errors")

        log_message(f"\nAnalysis complete! Results saved in: {args.output_dir}")
        log_message(f"Total GH annotations: {len(combined_results)}")
        log_message(f"Unique genomes: {combined_results['Genome'].nunique()}")
        log_message(f"GH families detected: {combined_results['GH_family'].nunique()}")
        if run_transporter_module and len(combined_transporter_results) > 0:
            log_message(
                f"Total transporter annotations: {len(combined_transporter_results)}")

        finalize_log(combined_results, genome_names, matrices,
                     combined_transporter_results if run_transporter_module else None)

    else:
        log_message("\nWARNING: No GH annotations found across all input genomes")
        log_message("No GH output files generated.")

        if run_transporter_module and len(combined_transporter_results) > 0:
            log_section("TRANSPORTER OUTPUT GENERATION")
            generate_transporter_annotations(combined_transporter_results, args.output_dir)
            generate_transporter_genome_summary(
                combined_transporter_results, args.output_dir, genome_names)

        finalize_log(pd.DataFrame(), genome_names, (None, None, None),
                     combined_transporter_results if run_transporter_module else None)

    print("=" * 80)
    print(f"Complete log saved to: "
          f"{os.path.join(args.output_dir, 'bifidoAnnotator_log.txt')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
