import numpy as np
import mne

def get_data(subjects, dataset, normalize = False):

    X_bci = []
    y_bci = []

    global bci_n_channels, bci_n_samples, bci_sfreq, n_classes_bci, bci_ch_names
    
    bci_ch_names = dataset[subjects[0]]['0train']['1'].ch_names
    bci_n_channels = len(bci_ch_names)
    bci_n_samples = dataset[subjects[0]]['0train']['1'].n_times
    bci_sfreq = 250

    for subject in dataset:
        for session in dataset[subject]:
            for trial in dataset[subject][session]:
                raw = dataset[subject][session][trial]
                events, event_id = mne.events_from_annotations(raw, verbose = False)
                epochs = mne.Epochs(raw, events, event_id, 2,6, baseline=None, preload=True, verbose = False)
                X_bci.append(epochs.pick('eeg').get_data(copy=False))
                y_bci.append(epochs.events[:, 2])
                

    X_bci = np.array(X_bci)
    y_bci = np.array(y_bci)

    X_bci_reshaped = X_bci.reshape(-1, X_bci.shape[2], X_bci.shape[3])
    y_bci_reshaped = y_bci.reshape(-1)

    # 2 = binary classification, left and right hand
    # 3 = left hand, right hand, feet
    # 4 = left hand, right hand, feet, tongue
    n_classes_bci = 2

    # Binary classification for left and right hand
    if(n_classes_bci == 2):
        mask = (y_bci_reshaped != 3) & (y_bci_reshaped != 4)
    elif(n_classes_bci == 3):
        mask = (y_bci_reshaped != 4) # add feet

    y_bci_reshaped = y_bci_reshaped[mask]
    X_bci_reshaped = X_bci_reshaped[mask]

    # Reset labels to 0 and 1 as expected by pytorch
    y_bci_reshaped = y_bci_reshaped - 1

    iir_params = dict(order=3, ftype="cheby1", rp=1, output="sos")
    low_cut_hz = 4.0
    high_cut_hz = 40.0

    # Apply band-pass filter
    X_bci_reshaped = mne.filter.filter_data(
        data=X_bci_reshaped,
        method="iir",
        iir_params=iir_params,
        sfreq=bci_sfreq,
        l_freq=low_cut_hz,
        h_freq=high_cut_hz,
        phase="forward",
        n_jobs=-1,
        verbose = False
    )

    # Apply z-score normalization on X
    if normalize:
        X_mean, X_std =  np.mean(X_bci_reshaped, axis=0),  np.std(X_bci_reshaped, axis=0)
        X_reshaped = (X_bci_reshaped - X_mean) / X_std

        return X_reshaped, y_bci_reshaped, X_mean, X_std
    else: 
        return X_bci_reshaped, y_bci_reshaped

