if __name__ == '__main__':
    from rta.read_in_data import DT
    import numpy as np
    import pandas as pd

    min_runs_no = 5

    index = DT.groupby('id')
    DT_stats = pd.DataFrame(dict(runs_no     = index.rt.count(),
                                 median_rt   = index.rt.median(),
                                 median_mass = index.pep_mass.median()))

    enough_runs = DT_stats.index[ DT_stats.runs_no >= min_runs_no ]

    DT = DT.loc[ DT.id.isin(enough_runs) ]
    DT = pd.merge(DT, DT_stats, left_on='id', right_index=True)


    x = DT.groupby('id').rt.size()
    DT = DT.loc[DT.id.isin(x[x >= 5].index)]

    DT.groupby('id')

    %%time
    x = DT.groupby('id').rt.size()
    x = x[x >= 5].index
    DT.loc[DT.id.isin(x)]

    sizes = DT.groupby('id', group_keys=False)['rt'].agg((np.mean, np.var, np.median))

    DT.groupby('id', group_keys=False).transform(np.median)
    DT.groupby('id')['rt'].size()
    DT.groupby('id', group_keys=False).apply(lambda x: len(x.rt))
    runs_no = DT.groupby('id', group_keys=False).apply(lambda x: len(x.rt))
    DT.iloc[list(runs_no >= 5)]
    medians = DT.groupby('id', group_keys=False).apply(lambda x: x.rt-np.median(x.rt))



    df = pd.DataFrame(dict(A = ('a', 'a', 'a', 'b', 'b', 'b', 'b'),
                           B = ( 1 ,  2 ,  3 ,  5 ,  5,   6,   9 ),
                           C = ( 1 ,  2 ,  3 ,  5 ,  5,   6,   9 )))

    df.groupby('A').transform(type)
    df.groupby('A')['B'].transform(type)
    df['median_dist'] = df.groupby('A')['B'].transform(lambda x: x - np.median(x))
    df.groupby('A').B.filter(lambda x: len(x))
    df.groupby('A').transform(lambda small_df: small_df['B'] )

    for g, d in df.groupby('A'):
        print(type(d))




    df.loc[[0,1]]


    data = pd.DataFrame(
        {'pid' : [1,1,1,2,2,3,3,3],
         'tag' : [23,45,62,24,45,34,25,62],
         })

    bytag = data.groupby('tag').aggregate(np.count_nonzero)

    data.groupby('tag').filter(lambda x: len(x) > 1)

    help(np.count_nonzero)

    tags = bytag[bytag.pid >= 2].index
    print(data[data['tag'].isin(tags)])
