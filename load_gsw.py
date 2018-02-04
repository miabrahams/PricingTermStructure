import pandas as pd
import numpy as np
from pandas.tseries.offsets import *

# Load Gurkaynak, Sack, Wright dataset.
# This data is extracted from here: https://www.federalreserve.gov/pubs/feds/2006/200628/200628abs.html
def load_gsw(filename: str, n_maturities: int):
    data = pd.read_excel(filename, parse_dates=[0])
    data = data.set_index('Date')
    data = data.resample('BM').last() # Convert to EOM observations
    data.index = data.index + DateOffset() # Add one day
    plot_dates = pd.DatetimeIndex(data.index).to_pydatetime() # pydatetime is best for matplotlib

    Beta0 = data['BETA0']
    Beta1 = data['BETA1']
    Beta2 = data['BETA2']
    Beta3 = data['BETA3']
    Tau1  = data['TAU1']
    Tau2  = data['TAU2']

    # Nelson, Svensson, Siegel yield curve parameterization
    def nss_yield(n):
        return Beta0 + Beta1 * (1 - np.exp(-n / Tau1)) / (n / Tau1) + \
               Beta2 * ((1 - np.exp(-n/Tau1))/(n/Tau1) - np.exp(-n/Tau1)) + \
               Beta3 * ((1 - np.exp(-n/Tau2))/(n/Tau2) - np.exp(-n/Tau2))

    # Compute yields
    rawYields = np.ndarray(shape=(data.shape[0], n_maturities))
    for mat in range(0, n_maturities):
        rawYields[:, mat] = nss_yield((mat+1)/12.0)

    return rawYields, plot_dates


