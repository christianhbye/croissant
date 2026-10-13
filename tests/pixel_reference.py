"""Slow scalar Earth reference using equal-area pixels and Astropy.

No Croissant or harmonic-transform code belongs here. Angles are radians,
frequencies MHz, and temperatures K. The celestial pixels have equal solid
angle, which cancels between numerator and denominator. Astropy includes
aberration and light deflection; refraction is disabled explicitly.
"""

from dataclasses import dataclass

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import AltAz, SkyCoord


def celestial_pixels(nside, frame="galactic"):
    """HEALPix RING pixel centres as colatitude, longitude, SkyCoord."""
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    directions = SkyCoord(
        phi * u.rad, (np.pi / 2 - theta) * u.rad, frame=frame
    )
    return theta, phi, directions


@dataclass
class PixelResult:
    """Independent full-beam, visible-sky and ground estimates."""

    temperature: np.ndarray
    sky_temperature: np.ndarray
    ground_fraction: np.ndarray


def integrate_pixels(sky, gain, visible, ground_temperature):
    """Integrate one snapshot, with (frequency, pixel) sky and gain.

    ``visible`` contains sky fractions in [0, 1]. The full-beam estimator
    is sum(B * [h*T + (1-h)*Tgnd]) / sum(B); the visible-sky estimator
    is sum(B*h*T) / sum(B*h). Neither denominator comes from the tested
    simulator. The ground temperature is scalar or one per frequency.
    """
    weighted = gain * visible
    full = np.sum(gain, axis=-1)
    unblocked = np.sum(weighted, axis=-1)
    sky_power = np.sum(weighted * sky, axis=-1)
    ground_fraction = np.sum(gain * (1 - visible), axis=-1) / full
    return PixelResult(
        sky_power / full + ground_fraction * ground_temperature,
        sky_power / unblocked,
        ground_fraction,
    )


def simulate_earth(
    sky_model,
    beam_model,
    horizon,
    freqs,
    times,
    location,
    *,
    nside,
    beam_rot=0.0,
    horizon_frame="topocentric",
    ground_temperature=0.0,
    frame="galactic",
):
    """Direct pixel sums at every UTC time, without a frozen frame.

    ``sky_model(theta, phi, freqs)`` and ``beam_model`` return arrays
    (frequency, pixel). The beam longitude is right-handed from its x
    axis; at zero rotation x is East and y is North. ``beam_rot`` is
    clockwise in degrees. ``horizon(azimuth)`` returns the minimum
    elevation in radians, with compass azimuth North through East. An
    attached horizon follows the antenna; a topocentric one stays fixed.
    """
    if horizon_frame not in {"beam", "topocentric"}:
        raise ValueError("horizon_frame must be 'beam' or 'topocentric'")
    theta, phi, directions = celestial_pixels(nside, frame)
    sky = sky_model(theta, phi, freqs)
    rotation = np.deg2rad(beam_rot)
    results = []
    for time in times:
        local = directions.transform_to(
            AltAz(obstime=time, location=location, pressure=0 * u.hPa)
        )
        az = local.az.to_value(u.rad)
        el = local.alt.to_value(u.rad)
        beam_phi = (np.pi / 2 + rotation - az) % (2 * np.pi)
        gain = beam_model(np.pi / 2 - el, beam_phi, freqs)
        mask_az = az if horizon_frame == "topocentric" else az - rotation
        visible = el >= horizon(mask_az % (2 * np.pi))
        results.append(
            integrate_pixels(sky, gain, visible, ground_temperature)
        )
    return PixelResult(
        np.stack([r.temperature for r in results]),
        np.stack([r.sky_temperature for r in results]),
        np.stack([r.ground_fraction for r in results]),
    )
