import numpy as np

def pyBetapart_sorensen(x):
    import warnings
    """
    Calculate multi-site Sørensen  similarity  index.
    x = 2d array, where rows are sites and columns are species.
    
    doi: 10.1098/rsbl.2006.0553
    doi: 10.1111/j.1600-0587.2012.00124
    https://github.com/cran/betapart
    https://stats.stackexchange.com/questions/280486
    
    """
    
    if x.shape[0] > x.shape[1]:
        warnings.warn('2d array has more sites than cells. Rows should be sites and columns cells.')
    x = x.astype(float)

    shared = x @ x.T
    not_shared = abs(shared - np.diag(shared))
    
    sumSi = np.diag(shared).sum() # species by site richness
    St = (x.sum(axis=0) > 0).sum() # regional species richness
    a = sumSi - St  # multi-site shared species term
    
    max_not_shared = np.maximum(not_shared, not_shared.T)
    min_not_shared = np.minimum(not_shared, not_shared.T)
    
    maxbibj = max_not_shared[np.tril_indices(max_not_shared.shape[0], k=-1)].sum()
    minbibj = min_not_shared[np.tril_indices(min_not_shared.shape[0], k=-1)].sum()
    
    return 1 - ((minbibj + maxbibj) / (minbibj + maxbibj +  (2 * a)))
