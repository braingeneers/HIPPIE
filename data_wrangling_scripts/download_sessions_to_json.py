#!/usr/bin/env python3
"""
Script to download Allen Institute Neuropixels Visual Coding sessions
and convert waveform and spike time data to JSON format.

Based on the original Jupyter notebook: getting_waveforms_and_spike_times.ipynb
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import argparse
import sys
from typing import Optional, Dict, Any


def setup_cache(output_dir: str = './local1/ecephys_cache_dir') -> EcephysProjectCache:
    """Initialize the Allen Institute data cache."""
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    return EcephysProjectCache.from_warehouse(manifest=manifest_path)


def filter_sessions(sessions: pd.DataFrame,
                   sex: Optional[str] = None,
                   genotype_filter: Optional[str] = None,
                   session_type: Optional[str] = None,
                   brain_region: Optional[str] = None) -> pd.DataFrame:
    """Filter sessions based on specified criteria."""
    filtered = sessions.copy()

    if sex:
        filtered = filtered[filtered.sex == sex]

    if genotype_filter:
        filtered = filtered[filtered.full_genotype.str.contains(genotype_filter, na=False)]

    if session_type:
        filtered = filtered[filtered.session_type == session_type]

    if brain_region:
        filtered = filtered[[brain_region in str(acronyms) for acronyms in
                           filtered.ecephys_structure_acronyms]]

    return filtered


def process_session(cache: EcephysProjectCache,
                   session_id: int,
                   sessions: pd.DataFrame,
                   output_dir: str = '.') -> bool:
    """Process a single session and save to JSON."""
    try:
        print(f"Processing session {session_id}...")

        # Get session data with permissive quality filters
        session = cache.get_session_data(
            session_id,
            isi_violations_maximum=np.inf,
            amplitude_cutoff_maximum=np.inf,
            presence_ratio_minimum=-np.inf
        )

        # Build JSON dictionary for this session
        json_dict = {}

        for unit_key in session.mean_waveforms.keys():
            json_dict[unit_key] = {
                "mean_waveform": session.mean_waveforms[unit_key].data[0].tolist(),
                "spike_times": session.spike_times[unit_key].tolist(),
                "session_type": sessions.loc[session_id].session_type,
                "specimen_id": str(sessions.loc[session_id].specimen_id),
                "age_in_days": str(sessions.loc[session_id].age_in_days),
                "full_genotype": sessions.loc[session_id].full_genotype,
                "ecephys_structure_acronym": session.units.loc[unit_key].ecephys_structure_acronym
            }

        # Save to JSON file
        output_file = os.path.join(output_dir, f'{session_id}.json')
        with open(output_file, "w") as outfile:
            json.dump(json_dict, outfile, indent=2)

        print(f"Saved {len(json_dict)} units from session {session_id} to {output_file}")
        return True

    except Exception as e:
        print(f"Error processing session {session_id}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download Allen Institute sessions and convert to JSON')
    parser.add_argument('--cache-dir', default='./local1/ecephys_cache_dir',
                       help='Directory for data cache')
    parser.add_argument('--output-dir', default='./session_json_files',
                       help='Directory for output JSON files')
    parser.add_argument('--max-sessions', type=int, default=None,
                       help='Maximum number of sessions to process (for testing)')
    parser.add_argument('--sex', choices=['M', 'F'],
                       help='Filter by animal sex')
    parser.add_argument('--genotype-filter',
                       help='Filter by genotype (partial string match)')
    parser.add_argument('--session-type',
                       help='Filter by session type')
    parser.add_argument('--brain-region',
                       help='Filter by brain region acronym')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Initialize cache
        print("Setting up Allen Institute data cache...")
        cache = setup_cache(args.cache_dir)

        # Get session table
        print("Retrieving session metadata...")
        sessions = cache.get_session_table()
        print(f"Found {len(sessions)} total sessions")

        # Apply filters if specified
        if any([args.sex, args.genotype_filter, args.session_type, args.brain_region]):
            sessions = filter_sessions(
                sessions,
                sex=args.sex,
                genotype_filter=args.genotype_filter,
                session_type=args.session_type,
                brain_region=args.brain_region
            )
            print(f"After filtering: {len(sessions)} sessions")

        # Limit sessions for testing if specified
        session_ids = sessions.index.tolist()
        if args.max_sessions:
            session_ids = session_ids[:args.max_sessions]
            print(f"Processing first {len(session_ids)} sessions for testing")

        # Process each session
        successful = 0
        failed = 0

        for i, session_id in enumerate(session_ids, 1):
            print(f"\n[{i}/{len(session_ids)}] ", end="")
            if process_session(cache, session_id, sessions, args.output_dir):
                successful += 1
            else:
                failed += 1

        print(f"\n\nProcessing complete!")
        print(f"Successfully processed: {successful} sessions")
        print(f"Failed: {failed} sessions")
        print(f"JSON files saved to: {args.output_dir}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()