"""
k_correction.py
Author: Benjamin Floyd

Converts an apparent magnitude to an absolute magnitude and applies a k-correction.
"""

import astropy.units as u
import numpy as np
from astropy.cosmology import default_cosmology
from astropy.table import Table
from synphot import SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D, ConstFlux1D


def k_correction(z, f_lambda, g_lambda_R, g_lambda_Q, R, Q):
    """
    Computes the K-correction for an observation through bandpass `R` at redshift `z` to a rest-frame measurement in
    bandpass `Q`.

    Parameters
    ----------
    z : float
        The redshift at which the the source was observed.
    f_lambda : Table or np.ndarray or SourceSpectrum
        Spectral energy distribution (SED) of flux. If not a `SourceSpectrum`, columns must be 'wavelength' and
        'f_lambda'.
    g_lambda_R, g_lambda_Q : str or u.Quantity or Table or np.ndarray or SourceSpectrum
        Spectral density of flux for the zero-magnitude standard source in bandpasses `R` and `Q` respectively. Input
        can be given in a variety of ways. An `Quantity` as input assumes a constant value over the range of the
        associated bandpass. If a Table-like object is given, columns must be 'wavelength' and 'f_lambda'. A
        `SourceSpectrum` object can be given directly for custom reference spectral distributions. There are also three
        built-in options that can be passed as strings.

        -`vega:` Uses the default Vega spectral distribution as provided by `synphot`. This can be checked with

            >>> from synphot.config import conf
            >>> print(conf.vega_file)

        - `ab:` Assumes a constant SED with
            :math:`g_\\nu(\\nu) = 3631 Jy = 3.631e-20 erg s^{{-1}} cm^{{-2}} Hz^{{-1}}`.
        - `st:` Assumes a constant SED with
            :math:`g_\\lambda(\\lambda) = 3.631e-9 erg s^{{-1}} cm^{{-2}} \\AA^{{-1}}`.

    R, Q : Table or np.ndarray or SpectralElement
        Spectral response curve (transmission) for bandpasses `R` and `Q` repectively. If a Table-like object is given,
        columns must be 'wavelength' and 'response'.

    Returns
    -------
    k_QR : float
        The K-correction for a source observed in bandpass `R` and in rest-frame bandpass `Q`.

    Examples
    --------
    Compute the K-correction for an elliptical galaxy at :math:`z = 0.3` observed in the SDSS g' filter corrected to the
    equivalent rest-frame SDSS u' filter.

    >>> import astropy.units as u
    >>> from synphot import SourceSpectrum, SpectralElement, units
    >>> sed = SourceSpectrum.from_file('SEDs/CWW/CWW_E_ext.sed', wave_unit=u.angstrom, flux_unit=units.FLAM)
    >>> sdss_u = SpectralElement.from_file('filter_curves/SDSS/filter_curves.fits', ext=1, wave_col='wavelength', flux_col='respt')
    >>> sdss_g = SpectralElement.from_file('filter_curves/SDSS/filter_curves.fits', ext=2, wave_col='wavelength', flux_col='respt')
    >>> k_corr = k_correction(z=0.3, f_lambda=sed, g_lambda_R='ab', g_lambda_Q='ab', R=sdss_g, Q=sdss_u)
    -0.46581583

    """

    # Check the types of the input parameters and cast them to synphot objects if necessary.
    if isinstance(f_lambda, (np.ndarray, Table)):
        f_lambda = SourceSpectrum(Empirical1D, points=f_lambda['wavelength'], lookup_table=f_lambda['f_lambda'])

    if isinstance(g_lambda_R, u.Quantity):
        g_lambda_R = SourceSpectrum(ConstFlux1D, amplitude=g_lambda_R)
    elif isinstance(g_lambda_R, (Table, np.ndarray)):
        g_lambda_R = SourceSpectrum(Empirical1D, points=g_lambda_R['wavelength'], lookup_table=g_lambda_R['f_lambda'])
    elif isinstance(g_lambda_R, str):
        try:
            g_lambda_R = BUILTIN_ZERO_PT[g_lambda_R.lower()]
        except KeyError:
            raise KeyError(f'{g_lambda_R.lower()} is not one of "vega", "ab", or "st".')

    if isinstance(g_lambda_Q, u.Quantity):
        g_lambda_Q = SourceSpectrum(ConstFlux1D, amplitude=g_lambda_Q)
    elif isinstance(g_lambda_Q, (Table, np.ndarray)):
        g_lambda_Q = SourceSpectrum(Empirical1D, points=g_lambda_Q['wavelength'], lookup_table=g_lambda_Q['f_lambda'])
    elif isinstance(g_lambda_Q, str):
        try:
            g_lambda_Q = BUILTIN_ZERO_PT[g_lambda_Q.lower()]
        except KeyError:
            raise KeyError(f'{g_lambda_Q.lower()} is not one of "vega", "ab", or "st".')

    if isinstance(R, (Table, np.ndarray)):
        R = SpectralElement(Empirical1D, points=R['wavelength'], lookup_table=R['response'])

    if isinstance(Q, (Table, np.ndarray)):
        Q = SpectralElement(Empirical1D, points=Q['wavelength'], lookup_table=Q['response'])

    # Take the input SED and make a redshifted copy to represent the observation-frame SED
    f_lambda_z0 = SourceSpectrum(f_lambda.model, z=0)
    f_lambda_z = SourceSpectrum(f_lambda.model, z=z)

    # Make Observation objects which convolve the SED with the filters
    obs_f_lambda_R = Observation(f_lambda_z, R)
    obs_f_lambda_z_Q = Observation(f_lambda_z0, Q)

    # Do the same for the zero-point reference SEDs
    obs_g_lambda_R_R = Observation(g_lambda_R, R)
    obs_g_lambda_Q_Q = Observation(g_lambda_Q, Q)

    # Compute the integrals
    integral_f_lambda_R = obs_f_lambda_R.integrate()
    integral_g_lambda_R_R = obs_g_lambda_R_R.integrate()
    integral_f_lambda_z_Q = obs_f_lambda_z_Q.integrate()
    integral_g_lambda_Q_Q = obs_g_lambda_Q_Q.integrate()

    # Compose the K-correction
    k_QR = -2.5 * np.log10(integral_f_lambda_R * integral_g_lambda_Q_Q /
                           (integral_g_lambda_R_R * integral_f_lambda_z_Q)).value

    return k_QR


# Vectorize the k_correction function
__k_correction_vect = np.vectorize(k_correction, otypes=[float],
                                   excluded=['f_lambda', 'g_lambda_R', 'g_lambda_Q', 'R', 'Q'])


def k_corr_abs_mag(apparent_mag, z, f_lambda_sed, zero_pt_obs_band, zero_pt_em_band, obs_filter, em_filter, cosmo=None):
    """
    Computes the absolute magnitude with K-correction applied given an apparent magnitude.

    Parameters
    ----------
    apparent_mag : float
        The apparent magnitude of source. Must use the same magnitude system as used in `zero_pt_obs_band`.
    z : float
        The redshift at which the the source was observed.
    f_lambda_sed : Table or np.ndarray or SourceSpectrum
        Spectral energy distribution (SED) of flux. If not a `SourceSpectrum`, columns must be 'wavelength' and
        'f_lambda'.
    zero_pt_obs_band : str or u.Quantity or Table or np.ndarray or SourceSpectrum
        Spectral density of flux for the zero-magnitude standard source in the observed-frame filter. See `k_correction`
        for more details.
    zero_pt_em_band : str or u.Quantity or Table or np.ndarray or SourceSpectrum
        Spectral density of flux for the zero-magnitude standard source in the emission-frame filter. See `k_correction`
        for more details.
    obs_filter : Table or np.ndarray or SpectralElement
        Spectral response curve (transmission) for observation-frame filter. See `k_correction` for more details.
    em_filter : Table or np.ndarray or SpectralElement
        Spectral response curve (transmission) for emission-frame filter. See `k_correction` for more details.
    cosmo : astropy.cosmology.core.Cosmology, optional
        Cosmology used to compute the distance modulus. If no cosmology is provided, the default cosmology as set in the
        astropy configuration.

    Returns
    -------
    absolute_mag : float
        The absolute magnitude of the object.

    See Also
    --------
    k_correction : K-correction computation
    k_corr_ap_mag : Apparent magnitude K-corrected from absolute magnitude.
    """

    # In case we weren't given a specific cosmology use the current default by Astropy
    if cosmo is None:
        cosmo = default_cosmology.get()

    # Compute the distance modulus
    dist_mod = cosmo.distmod(z).value

    # Compute the K-correction between the observed and emitted bandpasses
    k_corr = __k_correction_vect(z, f_lambda=f_lambda_sed, g_lambda_R=zero_pt_obs_band, g_lambda_Q=zero_pt_em_band,
                                 R=obs_filter, Q=em_filter)

    # Calculate the absolute magnitude in the requested emission band
    absolute_mag = apparent_mag - dist_mod - k_corr

    return absolute_mag


def k_corr_ap_mag(absolute_mag, z, f_lambda_sed, zero_pt_obs_band, zero_pt_em_band, obs_filter, em_filter, cosmo=None):
    """
    Computes the apparent magnitude with K-correction applied given an absolute magnitude.

    Parameters
    ----------
    absolute_mag : float
        The absolute magnitude of source. Must use the same magnitude system as used in `zero_pt_em_band`.
    z : float
        The redshift at which the the source was observed.
    f_lambda_sed : Table or np.ndarray or SourceSpectrum
        Spectral energy distribution (SED) of flux. If not a `SourceSpectrum`, columns must be 'wavelength' and
        'f_lambda'.
    zero_pt_obs_band : str or u.Quantity or Table or np.ndarray or SourceSpectrum
        Spectral density of flux for the zero-magnitude standard source in the observed-frame filter. See `k_correction`
        for more details.
    zero_pt_em_band : str or u.Quantity or Table or np.ndarray or SourceSpectrum
        Spectral density of flux for the zero-magnitude standard source in the emission-frame filter. See `k_correction`
        for more details.
    obs_filter : Table or np.ndarray or SpectralElement
        Spectral response curve (transmission) for observation-frame filter. See `k_correction` for more details.
    em_filter : Table or np.ndarray or SpectralElement
        Spectral response curve (transmission) for emission-frame filter. See `k_correction` for more details.
    cosmo : astropy.cosmology.core.Cosmology, optional
        Cosmology used to compute the distance modulus. If no cosmology is provided, the default cosmology as set in the
        astropy configuration.

    Returns
    -------
    apparent_mag : float
        The apparent magnitude of the object as observed at redshift `z`.

    See Also
    --------
    k_correction : K-correction computation
    k_corr_abs_mag : Absolute magnitude K-corrected from apparent magnitude.
    """

    # In case we weren't given a specific cosmology use the current default by Astropy
    if cosmo is None:
        cosmo = default_cosmology.get()

    # Compute the distance modulus
    dist_mod = cosmo.distmod(z).value

    # Compute the K-correction between the observed and emitted bandpasses
    k_corr = __k_correction_vect(z, f_lambda=f_lambda_sed, g_lambda_R=zero_pt_obs_band, g_lambda_Q=zero_pt_em_band,
                                 R=obs_filter, Q=em_filter)

    # Calculate the absolute magnitude in the requested emission band
    apparent_mag = absolute_mag + dist_mod + k_corr

    return apparent_mag


BUILTIN_ZERO_PT = {'vega': SourceSpectrum.from_vega(),
                   'ab': SourceSpectrum(ConstFlux1D, amplitude=1 * u.ABflux),
                   'st': SourceSpectrum(ConstFlux1D, amplitude=1 * u.STflux)}
