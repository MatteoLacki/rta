import numpy as np
import numpy.random as npr
import pandas as pd

def draw_peptide_rt(size=10000,
                    min_rt=16,
                    max_rt=180,
                    runs_no=10,
                    precision=.15):
    """Draw retention times fo peptides.
    
    Args:
        size (int): the number of peptides
    """
    rt = npr.random(size) * (max_rt - min_rt) + min_rt
    # in columns different runs
    rt_imprecise = npr.normal(loc=rt,
                              scale=precision,
                              size=(runs_no, size)).T
    rt_imprecise = pd.DataFrame(rt_imprecise)
    rt_imprecise.index = ["pep_{}".format(i) for i in range(size)]
    rt_imprecise = rt_imprecise.stack()
    rt_imprecise.index.names = ['peptide', 'run']
    rt_imprecise = rt_imprecise.reset_index(level=1)
    rt_imprecise.columns = ['run', 'rt']
    return rt_imprecise, rt

rt_imprecise, rt = draw_peptide_rt()
