import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile

from scipy import interpolate, signal
#import torch

from joblib import Parallel, delayed

from pynwb import NWBHDF5IO


class Neurocurator:
    def __init__(self):
        """
        Initializes a Hippie object with the given waveforms and spike times.
        waveforms: a pandas dataframe with columns representing waveform points and rows representing units
        spike_times: spikeData object from braingeneerspy
        sampling_rate: the sampling rate of the data
        metadata_obs: a pandas dataframe with metadata of the neurons
        isi_distribution: a pandas dataframe with the interspike interval distribution of the neurons
        waveform_embeddings: a pandas dataframe with the waveform embeddings of the neurons
        isi_distribution_embedding: a pandas dataframe with the interspike interval distribution embeddings of the neurons
        waveform_umap: a pandas dataframe with the waveform umap of the neurons
        isi_distribution_umap: a pandas dataframe with the interspike interval distribution umap of the neurons
        """
        self.waveforms = None
        self.spike_times_train = None
        self.templates = None
        self.sampling_rate = None
        self.metadata_obs = None
        self.isi_distribution = None
        self.acgs = None
        self.embeddings = None
        self.umap = None
        # self.connectivity_map = None

    def load_curation_file(self, qm_path):
        """Load data from a curation file."""
        if not os.path.exists(qm_path):
            raise FileNotFoundError(f"Curation file not found: {qm_path}")
            
        try:
            with zipfile.ZipFile(qm_path, "r") as f_zip:
                if "qm.npz" not in f_zip.namelist():
                    raise ValueError(f"Invalid curation file: missing qm.npz in {qm_path}")
                    
                qm = f_zip.open("qm.npz")
                data = np.load(qm, allow_pickle=True)
                
                # Validate required keys
                required_keys = ["train", "fs"]
                for key in required_keys:
                    if key not in data:
                        raise KeyError(f"Missing required data '{key}' in curation file")
                        
                # Extract data
                spike_times = data["train"].item()
                fs = data["fs"]
                train = [times / fs for _, times in spike_times.items()]
                
                config = data["config"].item() if "config" in data else None
                
                if "neuron_data" not in data:
                    raise KeyError("Missing required 'neuron_data' in curation file")
                neuron_data = data["neuron_data"].item()
                
        except zipfile.BadZipFile:
            raise ValueError(f"File is not a valid zip archive: {qm_path}")
        
        return train, neuron_data, config, fs

    def load_acqm(self, acqm_path):
        """
        Reads the acqm file and returns the waveforms and spike times of the neurons in the Hippie object.
        """
        train, neuron_data, config, fs = self.load_curation_file(acqm_path)
        # Sampling rate just in case
        self.sampling_rate = fs
        # Extract spike trains
        self.spike_times_train = train
        # Convert spike times to ms
        self.spike_times_train = [np.array(spike_times) * 1000 for spike_times in self.spike_times_train]
        # Extract waveforms
        self.waveforms = self.extract_waveforms(neuron_data)
        # Extract ISI distribution
        self.isi_distribution = self.compute_isi_distribution()
        # Extract autocorrelogram
        self.acgs = self.compute_autocorrelogram(train)
        # Extract position
        self.metadata_obs = self.extract_neuron_positions(neuron_data)

        #Validate data integrity
        self.validate_data_integrity()
        #Don't return anything, just set the attributes

        
    def extract_waveforms(self, neuron_data, n_datapoints=50):
        """
        Extracts the waveforms of the neurons in the Hippie object.
        """
        datapoints_before = int(n_datapoints / 5 * 2)
        datapoints_after = n_datapoints - datapoints_before

        neuron_dataframe = pd.DataFrame()
        neuron_array = []
        for neuron_id, neuron in neuron_data.items():
            # print(neuron_id,neuron)
            try:
                neuron_waveforms = neuron["waveforms"]
                neuron_flag = "waveforms"
                # Calculate mean waveform
                mean_waveform = np.mean(neuron_waveforms, axis=0)
                # Add mean waveform to dataframe
                neuron_array.append(mean_waveform)
            except:
                neuron_waveforms = neuron["template"]
                neuron_flag = "template"
                neuron_array.append(neuron_waveforms)


        if neuron_flag == "template":
            print("No waveforms, using template instead")

        # We would need to cut the neuron into X points 20 before the minimum and 30 after the minimum
        # We will use the minimum as the reference point
        neuron_cut = []
        for neuron in neuron_array:
            min_idx = np.argmin(neuron)
            try:
                neuron_cut.append(
                    neuron[min_idx - datapoints_before : min_idx + datapoints_after]
                )
            except:
                # If it goes out of bounds we will interpolate the waveform but now just fill with zeros
                neuron_cut.append(np.zeros(n_datapoints))

        neuron_dataframe = pd.DataFrame(neuron_cut)
        return neuron_dataframe

    def extract_neuron_positions(self, neuron_data):
        """
        Extracts the position of the neurons in the Hippie object.
        """
        position_array = []
        metadata = pd.DataFrame()
        for neuron_id, neuron in neuron_data.items():
            x, y = neuron["position"][0], neuron["position"][1]
            position_array.append([x, y])
        metadata["x"] = [x[0] for x in position_array]
        metadata["y"] = [x[1] for x in position_array]
        return metadata


    def compute_single_isi(self, spike_times):
        """
        Calculate interspike intervals (ISI) from an array of spike times.

        Parameters:
        spike_times (np.ndarray): A 1D numpy array of spike times (in seconds or ms).

        Returns:
        np.ndarray: A 1D array of interspike intervals.
        """
        # Ensure spike times is sorted
        spike_times.sort()
        # Ensure there are more than 2 spike times
        if len(spike_times) < 2:
            return np.array([])

        # Calculate the difference between successive spike times
        isi = np.diff(spike_times)
        # print(isi)
        return isi

    def compute_all_isis(self, spike_times_train):
        """
        Calculates the interspike intervals of the neurons in the Hippie object.
        Returns:
        isi: a list of numpy arrays with the interspike intervals of the neurons
        """
        isi = []
        for idx, neuron in enumerate(spike_times_train):
            if isinstance(neuron, np.ndarray):
                neuron = neuron.flatten()
            # If neuron is list, convert it to numpy array
            elif isinstance(neuron, list):
                neuron = np.array(neuron)
            # If neuron is not a numpy array, raise an error
            else:
                raise ValueError("neuron is not a numpy array")
            # Calculate ISI
            isi_unit = self.compute_single_isi(neuron)
            isi.append(isi_unit)
        return isi

    def compute_isi_distribution(self, time_window=100):
        """
        Extracts the interspike interval distribution of the neurons in the Hippie object.
        time_window: the time window in which to calculate the interspike interval distribution in milliseconds
        It will create a histogram with 1 ms bins
        """
        interspike_intervals_object = self.compute_all_isis(
            self.spike_times_train
        )

        def compute_hist(isi):
            if len(isi) == 0:
                return np.zeros(time_window)
            return np.histogram(isi[isi < time_window], bins=time_window)[0]
        
        isi_distribution = Parallel(n_jobs=-1)(
            delayed(compute_hist)(isi) for isi in interspike_intervals_object
        )
        return pd.DataFrame(isi_distribution)

    def compute_autocorrelogram(
        self,
        spike_trains,
        bin_size_ms=1,
        window_size_ms=100,
        normalize=False,
        remove_central_bin=True,
    ):
        """
        Extract autocorrelograms from a list of spike trains and return a DataFrame
        where each row is a neuron and each column is a time bin count.

        Parameters:
        -----------
        spike_trains : list of lists
            A list where each element is a list of spike times in milliseconds.
        bin_size_ms : float, default=1
            Size of each bin in milliseconds.
        window_size_ms : float, default=100
            Size of the window for computing autocorrelogram in milliseconds.
            The resulting histogram will span [-window_size_ms, window_size_ms].
        normalize : bool, default=False
            If True, normalize the autocorrelogram by dividing by the number of spikes.
        remove_central_bin : bool, default=True
            If True, remove the central bin (lag=0) to exclude self-comparisons.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame where:
            - Each row corresponds to a neuron (spike train)
            - Each column corresponds to a time bin
            - Column names are the bin centers in milliseconds
            - Values are the counts (or normalized counts) in each bin
        """
        # Calculate bin edges and centers
        bin_edges = np.arange(-window_size_ms, window_size_ms + bin_size_ms, bin_size_ms)
        bin_centers = bin_edges[:-1] + bin_size_ms / 2
        n_bins = len(bin_centers)

        def compute_single_acg(train):
            """Compute autocorrelogram for a single spike train efficiently."""
            # Convert to numpy array, flatten, and sort
            train = np.sort(np.asarray(train, dtype=np.float64).flatten())

            # Skip if too few spikes
            if len(train) < 2:
                return np.zeros(n_bins)

            n_spikes = len(train)
            differences = []

            # For each spike, find all spikes within the window using binary search
            for i in range(n_spikes):
                # Use searchsorted to efficiently find spikes within window
                # This is O(log n) instead of linear search
                upper_bound = train[i] + window_size_ms
                j_max = np.searchsorted(train, upper_bound, side='right')

                # Compute differences to spikes in range [i+1, j_max)
                if j_max > i + 1:
                    diffs = train[i+1:j_max] - train[i]
                    differences.extend(diffs)
                    differences.extend(-diffs)  # Add symmetric negative lags

            # Convert to numpy array
            differences = np.array(differences) if differences else np.array([])

            # Remove central bin (lag=0) if requested
            if remove_central_bin and len(differences) > 0:
                differences = differences[np.abs(differences) > 1e-10]

            # Create histogram
            if len(differences) > 0:
                counts, _ = np.histogram(differences, bins=bin_edges)
            else:
                counts = np.zeros(n_bins)

            # Normalize if requested
            if normalize and n_spikes > 0:
                counts = counts.astype(np.float64) / n_spikes

            return counts

        # Use parallel processing for better performance across spike trains
        all_acgs = Parallel(n_jobs=-1)(
            delayed(compute_single_acg)(train) for train in spike_trains
        )

        # Convert to numpy array then DataFrame
        all_acgs = np.array(all_acgs)

        # Create the DataFrame with appropriate column names
        column_names = [f"{x:.2f}" for x in bin_centers]
        result_df = pd.DataFrame(all_acgs, columns=column_names)

        return result_df

    def compute_cross_correlation(self, bt1, bt2, ccg_win=[-10, 10], t_lags_shift=0, bin_size=1):
        """
        Calculate cross-correlation between two binary spike trains.

        Parameters:
        -----------
        bt1 : numpy.ndarray
            First binary spike train.
        bt2 : numpy.ndarray
            Second binary spike train.
        ccg_win : list, default=[-10, 10]
            Window for cross-correlation in milliseconds [min, max].
        t_lags_shift : int, default=0
            Shift to apply to time lags.
        bin_size : float, default=1
            Size of each bin in milliseconds.

        Returns:
        --------
        tuple
            (counts, lags) - cross-correlation counts and corresponding time lags.
        """
        if np.all((np.array(ccg_win) / bin_size) % 1) != 0:
            raise ValueError("The window and shift must be multiples of the bin size")

        left_edge, right_edge = np.subtract(np.array(ccg_win) / bin_size, t_lags_shift)
        left_edge = int(left_edge)
        right_edge = int(right_edge)
        lags = np.arange(ccg_win[0], ccg_win[1] + bin_size, bin_size)

        pad_width = min(max(-left_edge, 0), max(right_edge, 0))
        bt2_pad = np.pad(bt2, pad_width=pad_width, mode="constant")

        cross_corr = signal.fftconvolve(bt2_pad, bt1[::-1], mode="valid")
        return np.round(cross_corr), lags


    def compute_waveform_features(self, waveform, mid=20, right_only=True, fs=20000.0):
        """
        From Sury's code
        measure the waveform features for both positive
        and negative spikes.
        waveform: a list or array of waveform
        mid: the index of the trough
        fs: sampling frequency
        return:
            a dictionary of features including
            left peak (index, value),
            right peak (index, value),
            trough to peak time (time in ms),
            fwhm (full width half maximum value in ms),
            fwhm_x (the left and right x values of the fwhm)
        """
        # 0. flip the waveform if it is positive
        if waveform[mid] > 0:
            waveform = -waveform
        # 1. measure the left and the right peak
        left = waveform[: mid + 1][::-1]
        right = waveform[mid:]
        ldiff = left[1:] - left[:-1]
        rdiff = right[1:] - right[:-1]
        lpeak_ind_list = np.where(ldiff < 0)[0]
        lpeak_ind, rpeak_ind = 0, len(waveform) - 1
        for ind in lpeak_ind_list:
            if waveform[mid - ind] > 0:
                lpeak_ind = mid - ind
                break
        lpeak_value = waveform[lpeak_ind]
        rpeak_ind_list = np.where(rdiff < 0)[0]
        for ind in rpeak_ind_list:
            if waveform[mid + ind] > 0:
                rpeak_ind = mid + ind
                break
        rpeak_value = waveform[rpeak_ind]
        lpeak = (lpeak_ind, lpeak_value)
        rpeak = (rpeak_ind, rpeak_value)
        # print(f"left peak {lpeak}, right peak {rpeak}")
        # 2. measure the trough to peak time (peak taken as the max peak either left or right)
        if not right_only:
            post_hyper = lpeak if lpeak_value > rpeak_value else rpeak
        else:
            post_hyper = rpeak
        trough_to_peak = abs(post_hyper[0] - mid) / (fs / 1000)  # make it in ms
        # print(f"{trough_to_peak} ms")
        # 3. measure the full width half maximum (FWHM) from the depolarization trough to the baseline
        half_amp = waveform[mid] / 2.0
        # print(f"half amplitude {half_amp}")
        # print(waveform[lpeak_ind: mid])
        # print(waveform[mid: rpeak_ind+1])
        xx_left = np.arange(lpeak_ind, mid + 1)
        xx_right = np.arange(mid, rpeak_ind + 1)
        # interpolate the left peak to trough waveform to find the half amplitude point
        fl = interpolate.interp1d(waveform[lpeak_ind : mid + 1], xx_left)
        fr = interpolate.interp1d(waveform[mid : rpeak_ind + 1], xx_right)

        # Check if half_amp is within the range of the original data
        if (
            np.min(waveform[lpeak_ind : mid + 1])
            <= half_amp
            <= np.max(waveform[lpeak_ind : mid + 1])
        ):
            inter_1 = fl(half_amp)
        else:
            inter_1 = None  # or some other appropriate value

        if (
            np.min(waveform[mid : rpeak_ind + 1])
            <= half_amp
            <= np.max(waveform[mid : rpeak_ind + 1])
        ):
            inter_2 = fr(half_amp)
        else:
            inter_2 = None  # or some other appropriate value

        fwhm_x = np.array([inter_1, inter_2])
        # Breaking :(
        try:
            fwhm = (abs(inter_2) - abs(inter_1)) / (fs / 1000)
        except:
            fwhm = 0
        # print(f"fwhm {fwhm} ms")
        features = {
            "lpeak": lpeak,
            "rpeak": rpeak,
            "trough_to_peak": trough_to_peak,
            "fwhm": fwhm,
            "fwhm_x": fwhm_x,
        }
        return features

    def compute_all_waveform_features(self):
        """
        Calculates the waveform features of the neurons in the Hippie object.

        """
        waveforms_lpeak = []
        waveforms_rpeak = []
        waveforms_trough_to_peak = []
        waveforms_fwhm = []
        waveforms_fwhm_x = []
        for idx, waveform in self.waveforms.iterrows():
            waveform = waveform.to_numpy()
            features = self.compute_waveform_features(waveform)
            waveforms_lpeak.append(features["lpeak"])
            waveforms_rpeak.append(features["rpeak"])
            waveforms_trough_to_peak.append(features["trough_to_peak"])
            waveforms_fwhm.append(features["fwhm"])
            waveforms_fwhm_x.append(features["fwhm_x"])

        self.metadata_obs["lpeak"] = waveforms_lpeak
        self.metadata_obs["rpeak"] = waveforms_rpeak
        self.metadata_obs["trough_to_peak"] = waveforms_trough_to_peak
        self.metadata_obs["fwhm"] = waveforms_fwhm
        self.metadata_obs["fwhm_x"] = waveforms_fwhm_x
        return self.metadata_obs

    def compute_firing_rate(self):
        """
        Calculates the most basic type of firing rate of the neurons in the Hippie object.
        """
        spike_train = self.spike_times_train
        firing_rate = []
        for idx, neuron in enumerate(spike_train):
            firing_rate.append(len(neuron) / (max(neuron) - min(neuron)))
        self.metadata_obs["firing_rate"] = firing_rate
        return self.metadata_obs
    
    def compute_minimum_isi(self):
        """
        Calculates the minimum interspike interval of the neurons in the Hippie object.
        """
        spike_train = self.spike_times_train
        minimum_isi = []
        for idx, neuron in enumerate(spike_train):
            isi = np.diff(neuron)
            minimum_isi.append(min(isi))
        self.metadata_obs["minimum_isi"] = minimum_isi
        return self.metadata_obs


    def set_experiment_condition(self, column_name, condition):
        """
        Sets the biological batch of the neurons in the Hippie object.
        """
        self.metadata_obs[column_name] = condition

    def validate_data_integrity(self):
        """Validate that all required data structures are properly loaded."""
        validation_results = {
            "waveforms": self.waveforms is not None and not self.waveforms.empty,
            "spike_times": self.spike_times_train is not None and len(self.spike_times_train) > 0,
            "sampling_rate": self.sampling_rate is not None and self.sampling_rate > 0,
            "metadata": self.metadata_obs is not None and not self.metadata_obs.empty,
            "isi_distribution": self.isi_distribution is not None and not self.isi_distribution.empty,
            "autocorrelograms": self.acgs is not None and not self.acgs.empty
        }
        
        missing = [k for k, v in validation_results.items() if not v]
        if missing:
            print(f"Warning: The following data structures are missing or empty: {', '.join(missing)}")
            
        return all(validation_results.values())
    
    def get_waveform_embeddings(self):
        return self.waveform_embeddings

    def get_isi_distribution_embedding(self):
        return self.isi_distribution_embedding

    def get_waveform_umap(self):
        return self.waveform_umap

    def get_isi_distribution_umap(self):
        return self.isi_distribution_umap

    def get_connectivity_map(self):
        """
        Returns the connectivity map of the neurons in the Hippie object. The connectivity map is a pandas dataframe with the following columns:
        - neuron1: the id of the first neuron
        - neuron2: the id of the second neuron
        - weight: the weight of the connection between the two neurons
        """
        return self.connectivity_map
    
    def calculate_average_sttc_SD(self, delta=10): #TODO: Change it to work with spike package from alex spaeth
        """
        Calculate average STTC of the neurons in the SpikeData object."""
        sttc = self.spike_times_train.spike_time_tilings(delt=delta)
        mean_sttc = np.mean(sttc, axis=0)
        self.metadata_obs["avg_sttc"] = mean_sttc
        return self.metadata_obs



    def calculate_firing_rate_SD(self, unit="Hz"): #TODO: Change it to work with spike package from alex spaeth
        """
        Calculates the firing rate of the neurons in the Hippie object.
        """
        # Calculate firing rate
        firing_rate = self.spike_times_train.rates(unit)
        self.metadata_obs["firing_rate"] = firing_rate
        return self.metadata_obs

    def calculate_metadata_SD(self):#TODO: Change it to work with spike package from alex spaeth
        """
        Calculates the metadata of the neurons in the Hippie object.
        """
        # Add column with average sttc
        self.calculate_average_sttc()
        # Add column with average firing rate
        self.calculate_firing_rate()
        # Add column with waveform features
        self.compute_all_waveform_features()
    
    def load_nwb_spike_times(self, nwb_path):
 
        
        with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
            nwb = io.read()

            #PyNWB exposes per-unit spike times via get_unit_spike_times(i) in seconds
            spikes_ms = []
            n_units = len(nwb.units.id.data)
            for i in range(n_units):
                st_sec = nwb.units.get_unit_spike_times(i)  #seconds (ragged vector)
                st_ms = np.asarray(st_sec, dtype=float) * 1000.0
                spikes_ms.append(st_ms)

        #fill object; downstream ISI/ACG code can run as-is
        self.spike_times_train = spikes_ms

    def load_nwb_waveforms(self, nwb_path, n_datapoints=50, candidates=("waveform_mean", "spike_waveforms")):
        #Candidates is the name of the columns to look for first

        #this functionensures all waveforms have the same number of points n, which is 50
        #since the other functions expect 50
        #if waveform is shorter, it pads with zeros at the end
        #if waveform is longer, it cuts around the minimum point
        #
        def cut_or_pad_to_n(wf, n=50):
            wf = np.asarray(wf, float).flatten()
            if wf.size == 0:
                return np.zeros(n, float)
            mid = int(np.argmin(wf))
            left = int(n * 2/5)
            right = n - left
            lo = max(0, mid - left)
            hi = min(wf.size, lo + n)
            seg = wf[lo:hi]
            if seg.size < n:
                seg = np.pad(seg, (0, n - seg.size))
            return seg

        #the function starts here by opening the nwb file
        with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
            nwb = io.read()
            #check if there is indeed a units table from where we can extract waveforms
            if nwb.units is None:
                self.waveforms = pd.DataFrame()
                return self.waveforms

            #check which of the candidate columns exists to use that one
            cols = set(nwb.units.colnames)
            key = None
            for k in candidates:
                if k in cols:
                    key = k
                    break

            #here it loops through the units id there is no waveform column it fills everything with zeros and terminates
            #if we have spike waveforms it takes all points and averages them (uses cut and pad function)
            #if it has waveform_mean it just uses that as the vector
            wf_rows = []
            n_units = len(nwb.units.id.data)
            for i in range(n_units):
                if key is None:
                    #no waveform column inside of units units = fills all with zeros
                    wf_rows.append(np.zeros(n_datapoints))
                    continue

                if key == "spike_waveforms":
                    #here it averages all the points and adds it
                    wfs = nwb.units.get_unit_spike_waveforms(i)
                    if wfs is None or len(wfs) == 0:
                        wf_rows.append(np.zeros(n_datapoints))
                    else:
                        mean_wf = np.mean(np.asarray(wfs, float), axis=0)
                        wf_rows.append(cut_or_pad_to_n(mean_wf, n=n_datapoints))
                else:
                    # if it already has the averge it just adds it
                    row = np.asarray(nwb.units[key][i], float)
                    wf_rows.append(cut_or_pad_to_n(row, n=n_datapoints))
        #define the universal waveform dataframe with the computed rows
        self.waveforms = pd.DataFrame(wf_rows)
        return self.waveforms
