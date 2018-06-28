D_all



annotated_cv, annotated_stats, run_cnts = \
    preprocess(annotated_all,
               min_runs_no,
               folds_no)


# variable to allign


assert all(vn in D_all.columns for vn in var_names), "Not all variable names are among the column names of the supplied data frame."
var_names in list(D_all.columns)



# def get_stats(D_all, min_runs_no, var_names,
#               pept_id='id'):
#     D_id = D_all.groupby(pept_id)


%%timeit
D_stats = D_id.agg({'rt': np.median, 
                    'mass': np.median,
                    'dt': np.median,
                    'run': ordered_str,
                    'id': len})



D_id = D_all.groupby('id')


def get_stats(D_all, 
              min_runs_no=5,
              pept_id='id',
              stats = {'rt':  np.median,
                       'dt':  np.median,
                       'mass':np.median}):
    
    D_id = D_all.groupby(pept_id)
    stats = 
    D_id





pd.unique(D_id.run)


D_id.run.first().groupby(level=0).size()


# add number to fold dictionary


%%timeit
D_id.run.agg(ordered_str)

D_id.run.agg(ordered_str)
D_id.run.value_counts()
D_id.run.value()