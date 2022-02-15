# input = pd series, sequence of observations
import numpy as np
import pandas as pd

def sym_CUSUM_filter(time_series, threshold):
    # time_series is pd series
    events, sPos, sNeg = [], 0, 0
    diff = time_series.diff()
    for i in diff.index[1:]:
        sPos = max(0,sPos+diff[i])
        sNeg = min(0, sNeg+diff[i])
        if sNeg < (-threshold):
            sNeg = 0
            events.append(i) # time that event occured
        elif sPos>h:
            sPos = 0
            events.append(i)
    return pd.DatetimeIndex(events)
