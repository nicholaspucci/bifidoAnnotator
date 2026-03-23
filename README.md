# bifidoAnnotator

A specialized bioinformatics pipeline for fine-grained annotation of bifidobacterial enzymes involved in human milk glycan (HMG) utilization, with publication-ready visualization capabilities.

## Overview

bifidoAnnotator provides an accurate, fast and user-friendly solution for hierarchical, homology-based protein cluster-level resolution of bifidobacterial enzymes (glycoside hydrolases, GHs) associated with HMG metabolism. Unlike broad GH family-level tools, bifidoAnnotator uses a manually curated database of 22,580 reference sequences organized into 108 functional protein clusters across 13 GH families, enabling identification of specific enzyme variants rather than just general GH family assignments.

## Features

- **Hierarchical annotation**: Multi-level GH classification with reference-specific thresholds
- **Transporter annotation** *(new in v2.0)*: Optional module that annotates substrate-specific transporters and links them to GH cluster assignments
- **Flexible input**: Single file or batch processing modes
- **Quality control**: Configurable coverage and bit score filtering
- **Comprehensive output**: Detailed tables, summary matrices, and publication-ready visualizations
- **Publication quality**: Vector graphics (PDF) and high-resolution raster (PNG) outputs
- **Detailed logging**: Complete analysis tracking

## Installation (python≥3.10 required)

### Conda Installation
```bash
# Install bifidoAnnotator with all dependencies (Recommended)
conda install -c bioconda -c conda-forge bifidoannotator

# Verify installation
bifidoAnnotator --help
```

### pip installation (Requires separate MMseqs2 installation)
```bash
# Install MMseqs2 first
conda install -c bioconda mmseqs2

# Then install bifidoAnnotator
pip install bifidoannotator

# Verify installation
bifidoAnnotator --help
```

### Manual installation and test run
```bash
# Create conda environment
conda create -n bifidoAnnotator python=3.10
conda activate bifidoAnnotator

# Install dependencies
conda install -c bioconda mmseqs2
pip install pandas seaborn matplotlib scipy

# Download bifidoAnnotator and its resources
git clone https://github.com/nicholaspucci/bifidoAnnotator.git
cd bifidoAnnotator

# Download and extract the reference databases from Zenodo (DOI: 10.5281/zenodo.19133752)
wget https://zenodo.org/records/19133752/files/bifDB_dir.tar.gz
tar -xvzf bifDB_dir.tar.gz -C database/

wget https://zenodo.org/records/19133752/files/bifTPDB_dir.tar.gz
tar -xvzf bifTPDB_dir.tar.gz -C database/

# Test run on 9 bifidobacterial genomes (with transporter module)
python script/run_bifidoAnnotator.py \
    -d Benchmark/ \
    --mapping_file database/mapping_file.tsv \
    --bifdb database/bifDB_dir/bifDB \
    -s Benchmark/genome_list_benchmark.txt \
    -o bifidoAnnotator_output \
    --annotations_file Benchmark/genome_info_benchmark.txt \
    --transporter_db database/bifTPDB_dir/bifTPDB \
    --transporter_mapping database/mapping_file_transporters.tsv
```

## Usage

### Quick Start (conda / pip package)
```bash
# Single genome analysis - GH database auto-downloaded on first run
bifidoAnnotator -i input_genome.faa -o output_directory

# Batch processing
bifidoAnnotator -d genome_directory -s sample_list.txt -o output_directory

# With genome metadata for enhanced heatmap annotation bars
bifidoAnnotator -i genome.faa --annotations_file metadata.tsv -o results
```

### Transporter module
```bash
# With transporter annotation (optional)
bifidoAnnotator -i genome.faa \
                --transporter_db /path/to/bifTPDB_dir/bifTPDB \
                --transporter_mapping /path/to/mapping_file_transporters.tsv \
                -o results
```

### Cluster heatmap row filtering
```bash
# Default: show clusters with HMG-utilization = Yes or Unknown
bifidoAnnotator -i genome.faa -o results

# Show only clusters with confirmed HMG-utilization
bifidoAnnotator -i genome.faa --hmg-only -o results

# Show all clusters regardless of HMG-utilization status
bifidoAnnotator -i genome.faa --all-genes -o results
```

### Advanced usage
```bash
# Custom threading and sensitivity
bifidoAnnotator -i genome.faa --threads 8 --sensitivity 7.5 -o results

# Manual figure sizing (width height in inches)
bifidoAnnotator -i genome.faa --gh-figsize 16 20 --cluster-figsize 18 22 -o results

# Red color scheme
bifidoAnnotator -i genome.faa -hc red -o results
```

## Input Files

### Required Inputs

1. **Protein FASTA files**: Amino acid sequences in FASTA format
   - Single file: Use `-i filename.faa`
   - Multiple files: Use `-d directory/` with `-s sample_list.txt`

2. **Mapping file** (`mapping_file.tsv`): TSV file with reference annotations containing:
   - `Protein_Name`: Reference sequence identifiers
   - `GH_family`: GH family assignments
   - `Enzyme`: Enzyme annotations
   - `Cluster_annotation`: Cluster assignments
   - `Validation_status`: Quality indicators
   - `GH_localization`: Subcellular localization
   - `HMG-utilization`: HMG-utilization capacity (Yes/No/Unknown)
   - `GH_family-F1_threshold`: Identity thresholds for GH family assignment
   - `GH-cluster threshold`: Identity thresholds for cluster assignment

### Optional Inputs

3. **Sample file**: Plain text file listing one genome name per line (required for batch mode `-d`)
   ```
   genome1
   genome2
   genome3
   ```

4. **Annotations file**: TSV file with genome metadata for enhanced heatmap annotation bars
   ```
   genome_name    species        isolation_source    other_metadata
   genome1        B. longum      infant              metadata1
   genome2        B. breve       adult               metadata2
   ```

5. **Transporter database** (`--transporter_db`): MMseqs2 transporter reference database (enables transporter module)

6. **Transporter mapping file** (`--transporter_mapping`): TSV file with transporter reference annotations containing:
   - `Transporter_gene`: Gene name of the transporter
   - `Reference_gene`: Reference gene identifier (`GeneName__Accession` format)
   - `Transporter_substrate_category`: Substrate category
   - `Transporter description`: Functional description
   - `Reference_species`: Species of origin
   - `Required by bifidoAnnotator clusters`: Semicolon-separated list of associated GH clusters

## Output Structure

```
output_directory/
├── bifidoAnnotator_log.txt
├── bifidoAnnotator_tables/
│   ├── detailed_annotations.tsv
│   ├── genome_summary.tsv
│   ├── gh_family_matrix.tsv
│   ├── enzyme_matrix.tsv
│   ├── cluster_matrix.tsv
│   ├── transporter_detailed_annotations.tsv   # only with --transporter_db
│   └── transporter_genome_summary.tsv         # only with --transporter_db
└── bifidoAnnotator_visualizations/
    ├── gh_family_heatmap.png
    ├── gh_family_heatmap.pdf
    ├── cluster_heatmap.png
    ├── cluster_heatmap.pdf
    ├── enzyme_heatmap.png
    └── enzyme_heatmap.pdf
```

## Output Files

### GH Annotation Tables
- **detailed_annotations.tsv**: Complete annotation results with sequence IDs, assignments, and quality metrics. Includes `Transporter_present` and `Transporter_ID` columns when the transporter module is enabled.
- **genome_summary.tsv**: Per-genome summary with copy numbers for each GH family/cluster. Includes transporter linkage columns when the transporter module is enabled.
- **Matrix files**: Wide-format presence/absence matrices for downstream analysis (GH family, enzyme, cluster)

### Transporter Annotation Tables (produced only with `--transporter_db`)
- **transporter_detailed_annotations.tsv**: One row per annotated transporter protein with gene name, substrate category, reference information, and alignment metrics
- **transporter_genome_summary.tsv**: Per-genome / per-transporter gene summary with copy numbers

### Visualizations
- **Heatmaps**: Hierarchically clustered heatmaps showing GH distribution patterns (PNG and PDF)
  - Automatic adaptive sizing based on data dimensions
  - Manual size control via `--gh-figsize`, `--cluster-figsize`, `--enzyme-figsize`
  - Enhanced versions with genome annotation bars (if `--annotations_file` provided)
  - **HMG-utilization row annotation bar** on cluster heatmap (colour-coded per cluster: Yes/No/Unknown)

## Parameters

### Core Parameters
| Parameter | Description |
|---|---|
| `-i, --input_file` | Single input FASTA file |
| `-d, --genome_directory` | Directory containing multiple FASTA files |
| `-s, --sample_file` | List of genome names for batch processing |
| `-o, --output_dir` | Output directory (default: `bifidoAnnotator_output`) |
| `--bifdb` | Path to MMseqs2 GH reference database (auto-downloaded on first run) |
| `--mapping_file` | GH reference annotation mapping file (bundled with package) |

### Optional Parameters
| Parameter | Default | Description |
|---|---|---|
| `--annotations_file` | — | Genome metadata TSV for enhanced heatmap annotation bars |
| `--threads` | 4 | Number of MMseqs2 threads |
| `--sensitivity` | 7.5 | MMseqs2 search sensitivity |
| `-hc, --heatmap_col` | `blue` | Color scheme: `blue` or `red` |
| `--gh-figsize W H` | auto | GH heatmap figure size in inches |
| `--cluster-figsize W H` | auto | Cluster heatmap figure size in inches |
| `--enzyme-figsize W H` | auto | Enzyme heatmap figure size in inches |

### Transporter Module Parameters
| Parameter | Description |
|---|---|
| `--transporter_db` | Path to MMseqs2 transporter reference database |
| `--transporter_mapping` | Path to transporter mapping TSV file |

Both arguments must be supplied together or not at all.

### Cluster Heatmap Row Filtering (mutually exclusive)
| Flag | Behaviour |
|---|---|
| *(default)* | Show clusters with HMG-utilization = Yes **or** Unknown |
| `--hmg-only` | Show only clusters with HMG-utilization = Yes |
| `--all-genes` | Show all clusters regardless of HMG-utilization status |

### Annotation Thresholds
- **GH annotation**: Coverage ≥ 50%, bit score ≥ 200, identity thresholds reference-specific from mapping file
- **Transporter annotation**: Coverage ≥ 50%, bit score ≥ 200, identity ≥ 80%

## Quality Control

The pipeline implements multiple quality control measures:

- **Coverage filtering**: Ensures alignments span sufficient sequence length (≥ 50%)
- **Bit score thresholds**: Filters low-confidence matches (≥ 200)
- **Reference-specific thresholds**: Uses optimized identity cutoffs per reference sequence
- **Hierarchical validation**: Applies different thresholds for family vs. cluster assignment
- **Transporter specificity**: Uses a stricter 80% identity threshold for transporter annotation

## Visualization Features

### Heatmap Characteristics
- **Hierarchical clustering**: Automatic genome and feature clustering (average linkage, Euclidean distance)
- **Discrete color scales**: Integer copy number representation with adaptive contrast per heatmap
- **Publication ready**: 300 dpi PNG + PDF vector output with professional typography
- **Adaptive sizing**: Automatic dimension calculation based on genome count and feature count
- **Annotation bars**: Per-genome metadata bars with distinct colour palettes
- **HMG row annotation bar**: Per-cluster HMG-utilization colour strip on cluster heatmap
- **Blue color scheme** by default; switch to red with `-hc red`

### Size Optimization
- Auto-calculates optimal dimensions based on:
  - Number of genomes (affects width)
  - Number of features (affects height)
  - Presence of annotation metadata
- Manual override available via `--gh-figsize`, `--cluster-figsize`, `--enzyme-figsize`

## Data Availability

The bifDB reference database (v2.0), bifTPDB transporter database, mapping files, and AIMS bifidobacterial proteomes are available on Zenodo:

**DOI: [10.5281/zenodo.19133752](https://doi.org/10.5281/zenodo.19133752)**

## Citation

Please cite this tool in your publications:

```
bifidoAnnotator: fine-grained annotation of bifidobacterial glycoside hydrolases for human milk glycan utilization
Authors: Nicholas Pucci & Daniel R. Mende
```

## Links

| Resource | URL |
|---|---|
| GitHub | https://github.com/nicholaspucci/bifidoAnnotator |
| PyPI | https://pypi.org/project/bifidoannotator/ |
| Bioconda | https://anaconda.org/bioconda/bifidoannotator |
| Database (Zenodo) | https://doi.org/10.5281/zenodo.19133752 |

## Support

For questions, bug reports, or feature requests, please open an issue on GitHub or contact the authors.

## License

MIT License - see LICENSE file for details

## Version History

- **v2.0**: Transporter annotation module; HMG-utilization row annotation bar on cluster heatmap; cluster row filtering flags (`--hmg-only`, `--hmg-unknown`, `--all-genes`); updated bifDB with subcellular localization, refined cluster annotations and outlier removal; new bifTPDB transporter database; updated Zenodo DOI
- **v1.0**: Complete pipeline integration with adaptive visualization, hierarchical annotation, quality control, and publication-ready outputs
