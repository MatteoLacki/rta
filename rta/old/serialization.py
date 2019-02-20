from rta.read.csvs import big_data
annotated_all, unlabelled_all = big_data()

unlabelled_all.to_msgpack('/Users/matteo/Projects/rta/data/unlabelled_all.msg')
annotated_all.to_msgpack('/Users/matteo/Projects/rta/data/annotated_all.msg')
unlabelled_all.to_feather('/Users/matteo/Projects/rta/data/annotated_all.feather')
unlabelled_all.to_feather()
store = pd.HDFStore('/Users/matteo/Projects/rta/data/annotated_all.h5')
store['unlabelled_all'] = unlabelled_all
store['annotated_all'] = annotated_all
