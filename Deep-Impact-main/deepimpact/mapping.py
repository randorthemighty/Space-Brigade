"""This module contains some useful mapping functions"""
import folium

__all__ = ["plot_circle"]


def plot_circle(lat, lon, radius, fmap=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    fmap: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> deepimpact.plot_circle(52.79, -2.95, 1e3, map=None)
    """

    if not fmap:
        fmap = folium.Map(location=[lat, lon], control_scale=True)

    folium.Circle([lat, lon], radius, fill=True, fillOpacity=0.6, **kwargs).add_to(fmap)

    return fmap
