======================
Classification Sandbox
======================

A bunch of routines for standard dumb classifiers that can serve as baselines.
Mostly the product of boredom.

Discrete (Multinomial) Naive Bayes
----------------------------------

This will fit a multinomial Naive Bayes model on observations which are 
assumed to be discrete. The input is a two-dimensional NumPy array 
(observations are rows, predictors are columns) and a one-dimensional
label vector. The elements can be integers, strings, or other Python
objects; anything that can be meaningfully compared for sorting and 
equality testing.

The following IPython session demonstrates most of its features 
(``fit_discrete_nb`` can take an optional third "pseudocounts" argument,
which indicates how many "fake observations" to add if a given feature
level/value has not been observed for a class; it defaults to 1).

::

    In [1]: data = np.loadtxt('agaricus-lepiota.data', dtype=np.dtype('S1'), delimiter=',')

    In [2]: labels = data[:,0] # Labels are the first column, 'e' for edible, 'p' for poison

    In [3]: data = data[:, 1:] # The rest are actual predictors

    In [4]: %run src/classification-sandbox/nbayes.py 

    In [5]: model = fit_discrete_nb(data, labels) # Fit the model

    In [6]: guesses = classify_discrete_nb(data, model)  # Test on training data, kind of silly

    In [7]: sum(classify_discrete_nb(data, model) == labels) / float(len(labels)) # How accurate were we?
    Out[7]: 0.94436238306253073

    In [8]: train_data = data[:6000]; train_labels = labels[:6000] # Split the data into train and test

    In [9]: test_data = data[6000:]; test_labels = labels[6000:]

    In [10]: model = fit_discrete_nb(train_data, train_labels) # Fit the model on only the training data

    In [11]: test_guesses = classify_discrete_nb(test_data, model) # Try to predict labels of test cases

    In [12]: sum(test_guesses == test_labels) / float(len(test_labels)) # A more honest assessment of accuracy
    Out[12]: 0.73728813559322037

    In [13]: test_guesses, posteriors = classify_discrete_nb(test_data, model, return_posterior=True)

    In [14]: posteriors
    Out[14]: 
    array([[  1.58396656e-06,   9.99998416e-01],
           [  5.04449409e-09,   9.99999995e-01],
           [  6.89993346e-08,   9.99999931e-01],
           ..., 
           [  5.57394259e-11,   1.00000000e+00],
           [  6.23636276e-07,   9.99999376e-01],
           [  1.10353548e-06,   9.99998896e-01]])

    In [15]: np.min(np.max(posteriors, axis=1)) # What was our least confident prediction?
    Out[15]: 0.536909514976459

