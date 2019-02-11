"""The class that applies a given model to each coordinate x."""


class Aligner(object):
    def __init__(self, M, *M_args, **M_kwds):
        """Constuctor.
        Args:
            M (callable): the type of model to be fitted to model y-x.
        """
        self.M = M
        self.m = {}

    def fit(self, X):
        """Fit all coordinate models of distances to reference.

        Args:
            X (pd.DataFrame): dataframe with (at least) columns r, x, y,
            where r - runs, x - observed, y - reference.
        """
        for r, Xr in X.groupby('run'):
            # y consists of reference RT
            self.m[r] = self.M(Xr.x.values,
                               Xr.y.values - Xr.x.values,
                               *self.M_args,
                               **self.M_kwds)
        # hence: we model the distance to the reference retention time.

    # to do: this might be not ordered according to the r values...
    def __call__(self, X):
        """Align the observations in X.

        Args:
            X (pd.DataFrame): dataframe with columns runs and x,
            where x are the values to be aligned.
            ATTENTION! sort X by r!!!
        """
        dx = np.zeros(X.shape[0])
        i = 0
        for r, Xr in X.groupby('r'):
            n = Xr.shape[0]
            dx[i:i+n] = self.m[r](Xr.x.values)
            i += n
        return X.x.values + dx

    #TODO: this should return error per r-basis?
    def error(self, X):
        """Evaluate the error on X.

        Args:
            X (pd.DataFrame): dataframe with (at least) columns r, x, y,
            where r - runs, x - data to be aligned, y - reference.
        """
        return np.median(np.abs(self(X) - X.y.values))

    def cv(self, X):
        """Cross-validate the model.

        Args:
            X (pd.DataFrame): dataframe with (at least) columns f, r, x, y, where f are folds, r - runs.
        Return:
            Average median test error across runs.
        """
        tot_err = 0
        f_vals = np.unique(X.f.values)
        for f in f_vals:
            X_train = X[X.f != f]
            X_test = X[X.f == f]
            m = Model(self.M, *self.M_args, **self.M_kwds)
            m.fit(X_train)
            tot_err += m.error(X_test)
        return tot_err / len(f_vals)
