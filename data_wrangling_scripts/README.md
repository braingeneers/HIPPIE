# Data Wrangling Scripts

This directory contains scripts for downloading and processing Allen Institute Neuropixels Visual Coding data.

## Setup

Create a conda environment with the required dependencies:

```bash
conda create -n allensdk python=3.9 -y
conda init
conda activate allensdk
pip install --no-input allensdk
pip install --no-input ipykernel
python -m ipykernel install --user --name=allensdk
```

## Scripts

### download_sessions_to_json.py

Converts the Jupyter notebook `getting_waveforms_and_spike_times.ipynb` into a standalone Python script that downloads all Allen Institute sessions and formats them into JSON files.

**Usage:**
```bash
python download_sessions_to_json.py [options]
```

**Options:**
- `--cache-dir`: Directory for Allen SDK data cache (default: `./local1/ecephys_cache_dir`)
- `--output-dir`: Directory for output JSON files (default: `./session_json_files`)
- `--max-sessions`: Maximum number of sessions to process (useful for testing)
- `--sex`: Filter by animal sex (`M` or `F`)
- `--genotype-filter`: Filter by genotype (partial string match)
- `--session-type`: Filter by session type
- `--brain-region`: Filter by brain region acronym

**Examples:**
```bash
# Download all sessions
python download_sessions_to_json.py

# Test with first 5 sessions only
python download_sessions_to_json.py --max-sessions 5

# Filter by male animals with Sst genotype
python download_sessions_to_json.py --sex M --genotype-filter Sst

# Download to custom directory
python download_sessions_to_json.py --output-dir ./my_json_files
```

**Output:**
Each session creates a JSON file named `{session_id}.json` containing:
- Unit IDs as keys
- For each unit:
  - `mean_waveform`: Average spike waveform shape
  - `spike_times`: All spike timestamps
  - `session_type`: Type of experimental session
  - `specimen_id`: Unique animal identifier
  - `age_in_days`: Animal age
  - `full_genotype`: Genetic background
  - `ecephys_structure_acronym`: Brain region