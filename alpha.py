# Alexandr (Sasha) Trubetskoy
# September 2018
# trub@uchicago.edu

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

from geopy import distance
from scipy.signal import savgol_filter
from mpl_toolkits.basemap import Basemap


wup = pd.read_excel('WUP2018-F12-Cities_Over_300K.xls', skiprows=16)


def get_latlon(cityname):
    assert (cityname in wup['Urban Agglomeration'].tolist()), "City not in data!"
    sel = wup[wup['Urban Agglomeration'] == cityname]
    if len(sel) > 1:
        largest = sel[2015].idxmax()
        sel = sel.loc[largest]
        print('There is more than one "{}". Assuming largest, which is in {}'
            .format(cityname, sel['Country or area']))
    origin = tuple(sel[['Latitude', 'Longitude']].values.flatten())
    return origin


def get_r(row, origin):
    destination = tuple(row[['Latitude', 'Longitude']].values.flatten())
    r = distance.distance(origin, destination).km
    return r


def get_zipf_mse(cityname, r):
    origin = get_latlon(cityname)

    wup['r'] = wup.apply(get_r, axis=1, args=(origin,))

    cities_within_radius = wup[wup['r'] <= r].copy()
    print('There are {} other cities within {} km of {}...'.format(len(cities_within_radius)-1, r, cityname))
    if len(cities_within_radius) == 1:
        return np.nan
    cities_within_radius['pop_rank'] = cities_within_radius[2015].rank(ascending=False)
    cities_within_radius['log_pop'] = (cities_within_radius[2015]*1e3).apply(np.log10)

    linreg = sm.ols(formula="log_pop ~ pop_rank", data=cities_within_radius).fit()
    mse = linreg.mse_resid
    print('\tMSE: {}'.format(mse))
    return mse


def get_mses(cityname):
    radii = list(range(500, 3000, 100))
    mses = [get_zipf_mse(cityname, r) for r in radii]
    return radii, mses


def plot_mses(radii, mses, cityname):
    fig, ax = plt.subplots()
    ax.scatter(x=radii, y=mses)
    ax.scatter(x=radii, y=savgol_filter(mses, 9, 2))
    ax.set_title(cityname)

    suffix = datetime.datetime.now().strftime('%d%b%Y_%H%M%S')
    plt.savefig(cityname+'_'+suffix+'.png')
    plt.close()
    pass


def get_elbow(rs, mses):
    THRESHOLD = 2e-3
    LOOKAHEAD = 0
    smooth = savgol_filter(mses, 9, 2)
    diffs = -np.diff(smooth)
    converges = diffs < THRESHOLD

    result = converges * list(range(len(converges)))
    # This checks to see if the next k values are also true
    for i in range(1, LOOKAHEAD+1):
        # Value becomes zero if i+k-th value is False
        result *= np.roll(converges, -i)

    # Find smallest nonzero index that passes.
    # Subtract 1 because elbow "begins" before the bend
    idx = np.min(result[np.nonzero(result)]) - 1
    return 1000*rs[idx]


def get_global_min(rs, mses):
    smooth = savgol_filter(mses, 9, 2)
    return 1000*rs[np.argmin(smooth)]


def plot_circles(radii, mses, cityname):
    # Find optimal circles
    elbow_r = get_elbow(radii, mses)
    glob_min_r = get_global_min(radii, mses)

    # Set up map with center at origin
    fig,ax = plt.subplots()
    lat_0, lon_0 = get_latlon(cityname)
    width = 6.2e6
    m = Basemap(
        width=width,
        height=width,
        projection='aeqd',
        lat_0=lat_0,
        lon_0=lon_0,
        resolution = 'l',
        ax=ax)
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()

    # Draw circles
    x,y = width/2, width/2
    x2,y2 = width/2, width/2 + elbow_r
    circle = plt.Circle((x, y), y2-y, color='red', fill=False)
    ax.add_patch(circle)

    x,y = width/2, width/2
    x2,y2 = width/2, width/2 + glob_min_r
    circle = plt.Circle((x, y), y2-y, color='gold', fill=False)
    ax.add_patch(circle)

    # Add title & annotations
    ax.set_title(cityname)
    ax.text(1e5, 5.5e5, 'Elbow method: {} km'.format(int(elbow_r/1e3)), 
        color='red', fontsize=10, bbox=dict(facecolor='white'))
    ax.text(1e5, 1.3e5, 'Global min: {} km'.format(int(glob_min_r/1e3)), 
        color='goldenrod', fontsize=10, bbox=dict(facecolor='white'))

    suffix = datetime.datetime.now().strftime('%d%b%Y_%H%M%S')
    plt.savefig(cityname+'_map_'+suffix+'.png', bbox_inches='tight')
    plt.close()
    pass


def go(cityname):
    radii, mses = get_mses(cityname)
    plot_mses(radii, mses, cityname)
    plot_circles(radii, mses, cityname)