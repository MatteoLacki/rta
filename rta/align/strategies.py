from rta.reference import stat_reference


def Tenzerize(X, n, a, stat='median'):
    """Perform a hunt for correct alignment."""
    for i in range(n):
        a.fit(X)
        x = a(X)
        X.rename(columns={'x':'x'+str(i), 'y':'y'+str(i)}, inplace=True)
        X['x'] = x
        X = stat_reference(X, stat)
    X.rename(columns={'x':'x'+str(n), 'y':'y'+str(n)}, inplace=True)
    return a, X
