from moabb.datasets import BNCI2014_001
import mne
import numpy as np


def get_bci_data(bci_subjects_excluded=[], test=False):
    """
    Get BCI data.

    This function retrieves BCI (Brain-Computer Interface) data from the BNCI2014_001 dataset.
    The data is preprocessed and returned as numpy arrays.

    Parameters:
    - bci_subjects_excluded (list): A list of subject numbers to exclude from the dataset. Default is an empty list.
    - test (bool): If True, retrieve test data. If False, retrieve training data. Default is False.

    Returns:
    - X_bci (ndarray): A 3-dimensional numpy array of shape (n_trials, n_channels, n_samples) containing the EEG data.
    - y_bci (ndarray): A 1-dimensional numpy array of shape (n_trials,) containing the corresponding labels.

    Note:
    - The BNCI2014_001 dataset is used to retrieve the data.
    - The MNE_DATA environment variable is set to "bciData" to specify the data directory.

    Example usage:
    >>> X, y = get_bci_data(bci_subjects_excluded=[2, 5], test=True)
    >>> X_test, y_test = get_bci_data(bci_subjects_excluded=[2, 5], test=True)

    """

    mne.set_config("MNE_DATA", "bciData")
    bci_subjects = [i for i in range(1, 10) if i not in bci_subjects_excluded]

    dataset = BNCI2014_001().get_data(subjects=bci_subjects)

    X_bci = []
    y_bci = []

    bci_sfreq = 250

    tmin = 0
    tmax = 639 / bci_sfreq

    for subject in dataset:
        session = "1test" if test else "0train"
        for trial in dataset[subject][session]:
            raw = dataset[subject][session][trial]
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            epochs = mne.Epochs(
                raw,
                events,
                event_id,
                tmin,
                tmax,
                baseline=None,
                preload=True,
                verbose=False,
            )
            X_bci.append(epochs.pick("eeg").get_data(copy=True))
            y_bci.append(epochs.events[:, 2])
    X_bci = np.array(X_bci)
    X_bci = X_bci.reshape(-1, X_bci.shape[2], X_bci.shape[3])
    y_bci = np.array(y_bci).reshape(-1)
    return X_bci, y_bci


def filter_bci_data(X, bci_sfreq=250):
    """
    Filter the BCI data using an IIR filter.

    Parameters:
        X (array-like): The input BCI data.
        bci_sfreq (float, optional): The sampling frequency of the BCI data. Defaults to 250.

    Returns:
        array-like: The filtered BCI data.

    """
    iir_params = dict(order=3, ftype="cheby1", rp=1, output="sos")
    low_cut_hz = 4.0
    high_cut_hz = 40.0

    mne.filter.filter_data(
        data=X,
        method="iir",
        iir_params=iir_params,
        sfreq=bci_sfreq,
        l_freq=low_cut_hz,
        h_freq=high_cut_hz,
        phase="forward",
        n_jobs=2,
    )

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X
