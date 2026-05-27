# Data Wrangling Scripts

Utilities for converting raw extracellular electrophysiology recordings — sourced from public archives such as **DANDI**, the **Allen Institute** Visual Coding NWB releases, and the **IBL Brain Wide Map** — into the per-modality CSV format HIPPIE consumes (`waveforms.csv`, `isi_dist.csv`, `acg.csv`, `labels.csv` / `metadata.csv`).

## Contents

| File | Purpose |
|------|---------|
| `neurocurator.py` | Core `Neurocurator` class: loads ACQM zips or NWB files, computes mean waveforms, ISI histograms, autocorrelograms, and per-unit shape features (FWHM, trough-to-peak, firing rate). |
| `acqm_to_csv_converter.ipynb` | End-to-end notebook: point it at an ACQM `.zip` and it writes a `<filename>_neurocurator_csv/` folder of HIPPIE-ready CSVs. |
| `allen_nwb_to_csv_converter.ipynb` | Downloads a single Allen Institute Visual Coding session by ID, runs it through `Neurocurator`, and writes the canonical CSV layout. Also serves as a template for any NWB file (DANDI dandisets). |
| `ibl_one_to_csv_converter.ipynb` | Downloads a single IBL Brain Wide Map insertion via the ONE API (Open Alyx public mirror, no credentials required), runs it through `Neurocurator`, and writes the canonical CSV layout. |

## Setup

The NWB / Allen path needs `allensdk` and `pynwb`. `allensdk` is finicky about Python versions, so we recommend an isolated env:

```bash
conda create -n allensdk python=3.9 -y
conda activate allensdk
pip install allensdk pynwb ipykernel
python -m ipykernel install --user --name=allensdk
```

The IBL path needs `ONE-api` and (optionally) `iblatlas` to map CCF region IDs to acronyms. Install both via the project's `[ibl]` extra:

```bash
pip install -e ".[ibl]"
```

Or directly:

```bash
pip install ONE-api iblatlas
```

The ACQM converter only needs the main HIPPIE env (`pynwb`, `scipy`, `joblib`, `pandas`).

## Output layout

Every script writes the same per-dataset structure HIPPIE expects:

```
<dataset_name>/
├── waveforms.csv      # n_units × 50 timepoints, trough-centered
├── isi_dist.csv       # n_units × 100 ms bins
├── acg.csv            # n_units × 200 ms bins (lag −100…+100 ms)
├── metadata.csv       # per-unit features (firing rate, FWHM, trough-to-peak, …)
└── session_summary.txt
```

Copy or symlink the resulting folder into `datasets_hippie/<name>/` (matching the dataset names listed in the top-level README) for HIPPIE to pick it up.

## Source-format → workflow map

| Source format | Where the paper's datasets come from | How to wrangle |
|---------------|--------------------------------------|----------------|
| **NWB (DANDI)** | Watson, Calvigioni, Ramachandran | `allen_nwb_to_csv_converter.ipynb` template — swap the Allen download cell for `dandi download dandiset:<id>`, then point `Neurocurator.load_nwb_spike_times()` + `load_nwb_waveforms()` at the local `.nwb`. |
| **NWB (Allen SDK)** | Allen Institute Visual Coding | `allen_nwb_to_csv_converter.ipynb` as-is — set `SESSION_ID` and run. |
| **IBL (ONE API)** | IBL Brain Wide Map | `ibl_one_to_csv_converter.ipynb` — set `EID` (insertion ID) and run. Downloads `spikes` / `clusters` / `channels` from the public Open Alyx mirror, optionally filters to IBL `good` units, and writes the canonical 5-CSV bundle. |
| **PhysMAP repository** | Juxtacellular S1, Lakunina A1, CellExplorer | The PhysMAP GitHub repo distributes these as pre-curated `.mat` / `.npz` files. Load them with `scipy.io.loadmat` / `numpy.load`, then construct the CSVs directly (no `Neurocurator` ACQM/NWB path needed — see notes per dataset below). |
| **C4 database** | Hausser, Hull, Lisberger | Download `.h5` files from [c4-database.com](https://www.c4-database.com); parse with custom scripts (see paper Methods §Per-unit feature extraction). Resulting per-unit mean waveforms + spike trains drop straight into the CSV layout. |
| **ACQM zip (Braingeneers)** | Braingeneers Mouse Organoid | `acqm_to_csv_converter.ipynb` — set `ACQM_ZIP_PATH`, run all cells. |

## Per-dataset reproduction recipes

These mirror the paper Methods §Datasets section. Quality filters and unit counts are taken from the paper; output directory names match the on-disk layout under `datasets_hippie/`.

### Cerebellar (C4 database — `https://www.c4-database.com`, cite Beau et al. 2024)

| Paper dataset | Output dir | Species | N (labeled) | Cell types |
|---------------|------------|---------|-------------|------------|
| Hausser | `hausser_cell_type` | mouse | 113 of 1,998 | GoC: 16, GrC: 9, MFB: 13, MLI: 15, PkC_cs: 25, PkC_ss: 35 |
| Hull | `hull_cell_type` | mouse | 103 | GoC: 2, MFB: 18, MLI: 13, PkC_cs: 34, PkC_ss: 36 |
| Lisberger | `lisberger_labeled_cell_type` | macaque (floccular complex) | 668 of 1,152 | GoC: 188, MFB: 86, MLI: 36, PkC_cs: 147, PkC_ss: 211 |


### PhysMAP-curated cortical datasets (from `https://github.com/EricKenjiLee/PhysMAP_Manuscript`)

| Paper dataset | Output dir | Citation | N | Classes |
|---------------|------------|----------|---|---------|
| Juxtacellular Mouse S1 | `juxtacellular_mouse_s1_area` | Yu et al. 2019 (juxtacellular micropipette) | 224 | L4 Exc: 58, L5 Exc: 43, L4 FS: 35, L5 FS: 19, SOM: 69 |
| Extracellular Mouse A1 (Lakunina) | `a1data_remove_undef` | Lakunina et al. 2020 (silicon probe) | 285 | Exc: 48, PV: 121, SST: 116 |
| CellExplorer (mouse VC + HPC) | `cellexplorer_cell_type` | Petersen 2021 / Siegle 2021 / Senzai 2019 (Neuropixels 1.0) | 430 | PV: 186, SST: 115, Pyramidal: 44, Axo-axonic: 35, Juxtacellular: 23, VIP: 14, VGAT: 13 |

A1 quality filter (per paper): ISI violation rate > 2 % excluded.

### DANDI Archive (NWB)

| Paper dataset | Dandiset | Citation | Recording tech | N | Classes |
|---------------|----------|----------|---------------|---|---------|
| Watson (rat frontal cortex) | [000041](https://dandiarchive.org/dandiset/000041) | Watson et al. 2016 | 64-site silicon probes | 221 (9 subjects) | Exc: 194, Inh: 27 |
| Calvigioni (mouse PFC) | [000473](https://dandiarchive.org/dandiset/000473) | Calvigioni et al. 2023 | Neuropixels | 9,213 (25 subjects) | Exc/RS: 7,859, Inh/FS: 1,354 |
| Ramachandran (rat S1) | [000955](https://dandiarchive.org/dandiset/000955) | Ramachandran et al. 2022 | NeuroNexus 32-ch | 134 (1 subject) | Inh: 105, Exc: 29 |

Cell-type labels live in the NWB `units` table for all three. Quality filters (from paper Methods):
- **Watson + Calvigioni:** minimum 100 spikes, ISI violations ≤ 0.5, presence ratio ≥ 0.5.
- **Ramachandran:** only labeled neurons are subset from the NWB; no explicit spike-count / ISI / presence-ratio filter is reported in the paper.

**Workflow:**
```bash
pip install dandi
dandi download https://dandiarchive.org/dandiset/000041   # or 000473, 000955
# then in allen_nwb_to_csv_converter.ipynb, replace the Allen download cell
# with the local .nwb path and run the rest.
```

### Allen Institute Visual Coding (AllenSDK)

- **Output dir:** `allen_scope_neuropixel_area`
- **N:** 61,781 units from 47 mice spanning 19 Allen CCF regions (visual cortex, hippocampus, thalamus/midbrain)
- **Quality filter:** ISI violations < 0.5, amplitude cutoff < 0.1, presence ratio > 0.9
- **Workflow:** `allen_nwb_to_csv_converter.ipynb` as-is, set `SESSION_ID` and iterate over the full session list. Cached SDK data goes to `./allen_cache/` (gitignored).

### IBL Brain Wide Map (ONE API)

- **Output dir:** (not shipped in this repo; suggested: `ibl_brainwide_map_area`)
- **N:** 62,993 neurons from 139 subjects and 698 probe insertions across 10 Cosmos-level brain regions
- **Quality filter:** IBL quality label `good` (ISI violations < 0.5, presence ratio ≥ 0.5, minimum 100 spikes) — `GOOD_ONLY = True` in the notebook (default).
- **Workflow:** open `ibl_one_to_csv_converter.ipynb`, set `EID` to the Brain Wide Map insertion you want, and run all cells. The notebook connects to the public Open Alyx mirror (`https://openalyx.internationalbrainlab.org`) anonymously. To iterate over the full BWM corpus, set `EID = None` first to list insertions, then loop the notebook (or extract its body into a Python script) over the returned list. See the [IBL ONE docs](https://int-brain-lab.github.io/iblenv/) for advanced query patterns.

### Braingeneers Mouse Organoid (ACQM, MaxOne HD-MEA)

- **Output dir:** `mouse_organoids_cell_line`
- **Source:** Braingeneers protosequences resource (`van2023protosequences`), three PSC lines (BRUCE4, KH2, ES-E14TG2a), 23–82 days in culture, MaxWell Biosystems MaxOne HD-MEAs
- **Workflow:** `acqm_to_csv_converter.ipynb`. Drop the ACQM `.zip` into `./data_acqm/` (gitignored), set `ACQM_ZIP_PATH`, run all cells.

## Notes

- `Neurocurator` auto-handles both the `waveform_mean` and `spike_waveforms` columns in NWB unit tables; recordings without waveform columns get zero-filled placeholders (see `load_nwb_waveforms`).
- Cached Allen SDK data lands in `./allen_cache/` by default — gitignored.
- Local ACQM zips go in `./data_acqm/` by convention — gitignored.
- Run outputs (`*_neurocurator_csv/`) are gitignored; copy or symlink the resulting CSVs into `datasets_hippie/<name>/` to use them in HIPPIE training.
- Quality-filter values above mirror what the paper reports; if you re-wrangle from scratch you should apply them at load time, not after the fact, since dropping units changes the ISI/ACG histograms.
