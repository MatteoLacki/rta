from math import sqrt, log as l, exp as e


def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0

class NoRootError(Exception):
    """No roots exception."""
    pass

def binomial_roots(A, B, C):
    """Compute the roots of binomial Ax**2+Bx+C.

    Numerically stable.
    W.H. Press, S.A.Teukolsky, W.T. Vetterling, B.P. Flannery,
    Numerical Recipes in C, Second Edition,
    Cambridge University Press,
    Australia 1992, pages 183-184.

    Args:
        A (float): quadratic coefficient.
        B (float): linear coefficient.
        C (float): constant coefficient.

    Returns:
        tuple: Two numbers: the smaller and the bigger root.

    Raises:
        NoRootError: I guess that this should be considered pathological, although: it's a matter of taste really.
    """
    assert A != 0.0, "The degree of the input must equal 2."
    delta = B**2 - 4*A*C
    if delta < 0.0:
        raise NoRootError("The determinant is negative: {}".format(delta))
    elif delta == 0.0:
        return -B/(2.0*A)
    else:
        Q = -(B + sign(B)*sqrt(delta))/2.0
        x1, x2 = Q/A, C/Q 
        return (x1, x2) if x1 <= x2 else (x2, x1)


def test_binomial_roots():
    """Test if the proper roots are being found."""
    x = binomial_roots(1, 0, -1)
    assert x[0] == -1.0 and x[1] == 1.0

    x = binomial_roots(1, 2, 1)
    assert x == -1.0

    x = binomial_roots(1, -8, 15)
    assert x[0] == 3, x[1] == 5