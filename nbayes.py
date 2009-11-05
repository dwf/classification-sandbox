from itertools import izip

import numpy as np

def discrete_to_int(arr, uniques=None):
    """
    Turn an array of discrete observations into a sorted array
    of unique values and an array of integer indices.

    If the second argument is specified, it uses this array 
    in place of unique(arr). arr need not contain every 
    element in uniques.
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise TypeError('array must be 1-dimensional')
    uniques = np.unique(arr)
    rows, cols = (arr == uniques[:, np.newaxis]).nonzero()
    return uniques, rows[np.argsort(cols)]

def fit_discrete_nb(features, labels, pseudocounts=1):
    """
    Fit a Naive Bayes model on discrete observations.
    """
    # Integerize the labels.
    unique_labels, int_labels = discrete_to_int(labels)
    
    # Some convenience variables for clarity
    nlabels = len(unique_labels)
    ndatapts, nfeatures = features.shape
    max_feat_levels = max(len(np.unique(feature)) for feature in features.T)
    
    feature_levels = [] 

    # Store the sufficient statistics in 3-dimensional array big enough to hold
    # them all. 
    #
    # Note that if there are many features with only a few levels and
    # one or two with a lot, this will lead to a horrible waste of memory. A 
    # sparse data structure would be more appropriate in general.
    stats = np.empty((nlabels, nfeatures, max_feat_levels), dtype=float)
    
    # Flag off everything as NaN so that they aren't spurious entries 
    # (and we can use nansum() if necessary).
    stats[...] = np.nan

    for f_index, feature in enumerate(features.T):

        # Get integers from the discrete feature levels.
        thisfeat_levels, indices = discrete_to_int(feature)
        
        # Append the actual levels of the features so that we know all of
        # the levels (and their order) at test time.
        feature_levels.append(thisfeat_levels)

        for label in range(nlabels):

            # Count the number of cases with each level, turn into a float
            # so we can normalize it (from __future__ import division doesn't
            # affect NumPy, does it?).
            counts = np.atleast_1d(np.bincount(indices[int_labels == label]))

            # If we haven't observed this feature level for any data cases, 
            # add pseudocounts to smooth over it.
            counts[counts == 0] = pseudocounts
            counts = np.atleast_1d(np.float64(counts))
            counts /= float(counts.sum())
            print counts
            stats[label, f_index, 0:len(counts)] = counts
            
    return stats, feature_levels, unique_labels, \
        np.bincount(int_labels) / float(len(int_labels))

def classify_discrete_nb(test_features, model, return_posterior=False):
    """
    Given the output of fit_discrete_nb and some test features, classify 
    a test set.
    """
    test_features = np.asarray(test_features)
    stats, feature_levels, unique_labels, prior = model
    nlabels, nfeatures, nlevels = stats.shape
    log_prior = np.log(prior)
    log_stats = np.log(stats)
    numeric_features = np.empty(test_features.shape, dtype=int)

    feat_and_levels = izip(test_features.T, feature_levels)
    for f_index, (test_feature, levels) in enumerate(feat_and_levels):
        numeric_features[:, f_index] = discrete_to_int(test_feature, levels)[1]

    # Two index arrays that will pull out all the probabilities.
    idx_f, idx_n = np.broadcast_arrays(np.arange(nfeatures)[np.newaxis, :],
                                       numeric_features)
    
    joints = np.nansum(log_stats[:, idx_f, idx_n], axis=-1).transpose()
    joints += log_prior[np.newaxis, :]
    
    guesses = np.argmax(joints, axis=1)
    
    print np.isnan(joints).sum()

    if return_posterior:
        # Do it in-place since we don't actually need this array anymore.
        np.exp(joints, joints)
        joints /= joints.sum(axis=1)[:, np.newaxis]
        return unique_labels[guesses], joints
    
    else:
        return unique_labels[guesses]
