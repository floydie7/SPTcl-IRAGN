"""
custom_math.py
Author: Benjamin Floyd

A small package to store and test custom mathematical functions
"""

import numpy as np


def trap_weight(y, x=None, dx=1.0, axis=-1, weight=None):
    """
    Trapezoidal integration with weights on each subinterval. Mostly based on numpy.trapz.

    Integrate `y` (`x`) along given axis with each subinterval weighted by `weight`.

    :param y:
        Input array to integrate
    :type y: array_like

    :param x:
        The sample points corresponding to the `y` values. If `x` is None, the sample points are assumed to be evenly
        spaced `dx` apart. The default is None.
    :type x: array_like, optional

    :param dx:
        The spacing between sample points when `x` is None. The default is 1.
    :type dx: scalar, optional

    :param axis:
        The axis along which to integrate.
    :type axis: int, optional

    :param weight:
        The weights corresponding to the subintervals of width `dx` or spaced according to `x`. If `weight` is None,
        this function will fall back to numpy.trapz. The default is None.
    :type weight: array_like, optional


    :return trapz_weight:
        Definite integral as approximated by the trapezoidal rule with each interval appropriately weighted.
    :rtype: float
    """

    # First check if weight is supplied. If not, fall back to numpy.trapz.
    if weight is None:
        return np.trapz(y, x, dx, axis)

    # The following lines are taken from numpy.trapz
    # First, convert `y` to ndarray via numpy.asanyarray
    y = np.asanyarray(y)

    # If `x` is not provided, set the spacing to be `dx`.
    if x is None:
        d = dx
    else:
        # Convert `x` to ndarray via numpy.asanyarray
        x = np.asanyarray(x)

        if x.ndim == 1:  # If `x` is 1D then simply take the first difference to find the spacing.
            d = np.diff(x)

            # Reshape to correct shape
            shape = [1]*y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)

        else:  # Find the spacing along the given axis.
            d = np.diff(x, axis=axis)

    # Create slices to evaluate `y` on
    nd = y.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    # Here we modify the original code to hold off the summation until we have incorporated the weights
    weight = np.asanyarray(weight)

    try:
        sub_integral = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
        integral = np.sum(sub_integral * weight, axis)

    except ValueError:
        # Operations didn't work, force casting to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        weight = np.asarray(weight)

        sub_integral = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
        integral = np.add.reduce(sub_integral * weight, axis)

    return integral


def bootstrap(data, bootnum=100, samples=None, bootfunc=None):
    if samples is None:
        samples = data.shape[0]

    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")

    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
        # test number of outputs from bootfunc, avoid single outputs which are
        # array-like
        try:
            resultdims = (bootnum, len(bootfunc(data)))
        except TypeError:
            resultdims = (bootnum,)

    # create empty boot array
    boot = np.empty(resultdims, dtype=data.dtype)

    for i in range(bootnum):
        bootarr = np.random.randint(low=0, high=data.shape[0], size=samples)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr])

    return boot


def rchisq(ydata, ymodel, n_parameters=2, stdev=None, both=False):
    """
    Computes the reduced chi-squared goodness of fit statistic.

    Parameters
    ----------
    ydata : array_like
        The data being fit.
    ymodel : array_like
        Model predictions y = f(x).
    n_parameters : int, optional
        Number of free parameters in model. Defaults to a two parameter model
    stdev : array_like, optional
        Error associated with the data. If not specified, we will compute the least-squares statistic instead.
    both : bool, optional
        If `True` returns both the chi-squared statistic and the reduced chi-squared statistic. Defaults to `False`.

    Returns
    -------
    chi2 : float
        Chi-squared statistic, only returned if `both` is `True`.
    reduced_chi2 : float
        Reduced Chi-squared statistic.
    """

    # Chi-square statistic
    if stdev is None:
        chisq = np.sum((ydata - ymodel) ** 2)
    else:
        chisq = np.sum(((ydata - ymodel) / stdev) ** 2)

    # Number of degrees of freedom assuming 2 free parameters
    nu = ydata.size - 1 - n_parameters

    if both:
        return chisq, chisq / nu

    return chisq / nu
