"""
Generate csv and adjlist file for tutorial.
"""

import numpy as np, pandas as pd, networkx as nx, sys
sys.path.append('../../src/')
import networkinference.utils as nu
import networkinference as ni

# generate data
Y, X, W, A = nu.FakeData.tsls(n=1000, network='RGG')

# stats
ni.core.sumstats(A)
tsls = ni.TSLS(Y, X, W, A)
print(tsls.estimate)

# save network in adjlist format
nx.write_adjlist(A, 'network.adjlist')

# save node-level data in csv
data = pd.DataFrame(np.vstack([Y, X[:,2]]).T) 
data.to_csv('node_data.csv', header=['Y','X'], index=False)
