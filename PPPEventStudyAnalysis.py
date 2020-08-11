import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
from collections import defaultdict
from collections import OrderedDict
from math import sqrt
import timeit
from tqdm import tqdm

"""
The majority of the code used to perform the actual calculations of abnormal return is from

http://esocialtrader.com/event-studies/

The inital manipulation of datasets is custom and rather inefficient

This code takes ~45 minutes to run (on my computer) with a dataset of similar size to the original, 355 days x 3202 companies

Your timing results may vary
"""

start_time = timeit.default_timer()



crsp_data = (
    'C:/Users/blkel/Dropbox/BDC Research/Event Study Analysis/data/data_for_Brian_crsp.csv')
crsp_df = pd.read_csv(crsp_data, usecols = ["date", "permno", "ret"])

f_f_data = (
    'C:/Users/blkel/Dropbox/BDC Research/Event Study Analysis/data/F-F_Research_Data_Factors_daily.csv')
f_f_df = pd.read_csv(f_f_data)

ppp_data = (
    'C:/Users/blkel/Dropbox/BDC Research/Event Study Analysis/data/data_for_Brian_ppp.csv')
ppp_df = pd.read_csv(ppp_data)

ppp_df.dropna(subset = ["ppp_sec_filing_date1"], inplace = True)

ppp_df = ppp_df.drop(columns = ["no_gkvey", "no_cik_from_compustat","ind_borrow_ppp"])

ppp_df["ppp_sec_filing_date1"] = pd.to_datetime(ppp_df["ppp_sec_filing_date1"], format='%m/%d/%Y')

ppp_df["ppp_sec_filing_date2"] = pd.to_datetime(ppp_df["ppp_sec_filing_date2"], format='%d-%b-%y')

event_permno_list = []

events_dict = {}
for permno, data in ppp_df.groupby("permno"):
    events_dict[permno] = data.iloc[0]["ppp_sec_filing_date1"]
    event_permno_list.append(int(permno))


f_f_array = f_f_df.values
f_f_values = f_f_array[24395:-2]
f_f_edited = pd.DataFrame(f_f_values, columns=["Date", "Mkt-RF", "SMB", "HML", "RF" ])
f_f_edited["Date"] = pd.to_datetime(f_f_edited["Date"], format='%Y%m%d')
f_f_edited["Mkt-RF"] = f_f_edited["Mkt-RF"].astype(float)
f_f_edited["RF"] = f_f_edited["RF"].astype(float)
f_f_edited["Mkt"] = f_f_edited["Mkt-RF"] - f_f_edited["RF"]
f_f_edited = f_f_edited.drop(columns = ["SMB", "HML", "Mkt-RF", "RF"])
f_f_edited["Mkt"] = f_f_edited["Mkt"] / 100

crsp_df_new = crsp_df.rename(columns={"date":"Date"})

crsp_df_new["Date"] = pd.to_datetime(crsp_df_new["Date"], format='%Y-%m-%d')

unique_dates = crsp_df_new["Date"].unique()
unique_permnos = crsp_df_new["permno"].unique()

crsp_df_new = crsp_df_new.drop_duplicates(subset=["Date", "permno"], ignore_index = True)

ret_df = crsp_df_new.pivot(index = "Date", columns = "permno")["ret"].reset_index()
ret_df.columns.name = None


data_ret = pd.merge(ret_df, f_f_edited, on="Date", how = "outer")
data_ret = data_ret.set_index("Date")
data_ret.columns.name = None
data_ret.drop(data_ret.index[:150], inplace=True)
data_non_ppp = data_ret.copy()
data_ppp = data_ret.copy()


for permno in event_permno_list:
    data_non_ppp = data_non_ppp.drop(columns = permno, errors = 'ignore')

ready_permnos = data_non_ppp.columns.values.tolist()


daily_diff = 0.02

# Events col is permnos that had events
events_col = ready_permnos[:]
events_index = data_non_ppp.index
# Making a new dataframe, row is date, col
data_events = pd.DataFrame(index=events_index, columns=events_col)

for i in events_col:
    data_events[i] = np.where((data_non_ppp[i] - data_non_ppp['Mkt']) > daily_diff, 1, np.where((data_non_ppp[i] - data_non_ppp['Mkt']) < -daily_diff, -1, np.nan))

print("Evaluation Complete")

L1 = 30
window = 20

pos_dict = defaultdict(dict)
neg_dict = defaultdict(dict)

for s in tqdm(events_col):
    pos_event_dates = data_events[s][data_events[s] == 1].index.tolist()
    neg_event_dates = data_events[s][data_events[s] == -1].index.tolist()

    pos_dict_s = defaultdict(dict)
    neg_dict_s = defaultdict(dict)

    for pos_event in pos_event_dates:
        date_loc = data_non_ppp.index.get_loc(pos_event)
        date_loc = date_loc - window

        if date_loc > L1 and date_loc <= len(data_non_ppp) - (2*window+1):
            index_range = (2*window) + 1

            pos_dict_s_event = OrderedDict()
            for d in range(index_range):
                date_loc2 = date_loc + d

                u_i = data_non_ppp[s][date_loc2 - L1: date_loc2 - 1].mean()
                u_m = data_non_ppp['Mkt'][date_loc2 - L1: date_loc2 - 1].mean()
                R_i = data_non_ppp.iloc[date_loc2, data_non_ppp.columns.get_loc(s)]
                R_m = data_non_ppp.iloc[date_loc2, data_non_ppp.columns.get_loc('Mkt')]
                beta_i = ((R_i - u_i) * (R_m - u_m)) / (R_m - u_m) ** 2
                alpha_i = u_i - (beta_i * u_m)
                var_err = (1 / (L1 - 2)) * (R_i - alpha_i - (beta_i * R_m)) ** 2
                AR_i = R_i - alpha_i - (beta_i * R_m)

                pos_dict_s_event[date_loc2] = AR_i

            pos_dict_s[pos_event] = pos_dict_s_event

    pos_dict[s] = pos_dict_s


    for neg_event in neg_event_dates:
        date_loc = data_non_ppp.index.get_loc(neg_event)
        date_loc = date_loc - window

        if date_loc > L1 and date_loc <= len(data_non_ppp) - (2*window+1):
            index_range = (2*window) + 1

            neg_dict_s_event = OrderedDict()
            for d in range(index_range):
                date_loc2 = date_loc + d

                u_i = data_non_ppp[s][date_loc2 - L1: date_loc2 - 1].mean()
                u_m = data_non_ppp['Mkt'][date_loc2 - L1: date_loc2 - 1].mean()
                R_i = data_non_ppp.iloc[date_loc2, data_non_ppp.columns.get_loc(s)]
                R_m = data_non_ppp.iloc[date_loc2, data_non_ppp.columns.get_loc('Mkt')]
                beta_i = ((R_i - u_i) * (R_m - u_m)) / (R_m - u_m) ** 2
                alpha_i = u_i - (beta_i * u_m)
                var_err = (1 / (L1 - 2)) * (R_i - alpha_i - (beta_i * R_m)) ** 2
                AR_i = R_i - alpha_i - (beta_i * R_m)

                neg_dict_s_event[date_loc2] = AR_i

            neg_dict_s[neg_event] = neg_dict_s_event

    neg_dict[s] = neg_dict_s



abret_col = ready_permnos[:]
abret_col.remove("Mkt")
abret_index = range(-window, window + 1)
pos_data_abret = pd.DataFrame(index=abret_index, columns=abret_col)
neg_data_abret = pd.DataFrame(index=abret_index, columns=abret_col)

for h in abret_col:
    if h in pos_dict.keys():
        for z in abret_index:
            pos_data_abret[h][z] = np.mean([list(x.values())[z + window] for x in pos_dict[h].values()])

for f in abret_col:
    if f in neg_dict.keys():
        for v in abret_index:
            neg_data_abret[f][v] = np.mean([list(x.values())[v + window] for x in neg_dict[f].values()])

print("Done Setting Up for Plots")

pos_CAR = pos_data_abret.cumsum()
neg_CAR = neg_data_abret.cumsum()

pos_CAR['SUM'] = pos_CAR.sum(axis=1)/11.86
neg_CAR['SUM'] = neg_CAR.sum(axis=1)/11.86


"""

Now we repeat this process for the companies which DID recieve PPP loans

"""




for permno in data_ppp.columns:
    if permno not in event_permno_list:
        if permno != "Mkt":
            data_ppp = data_ppp.drop(columns = permno, errors = 'ignore')

ready_permnos_ppp = data_ppp.columns.values.tolist()


daily_diff = 0.02

events_col_ppp = ready_permnos_ppp[:]
events_index_ppp = data_ppp.index
# Making a new dataframe, row is date, col
data_events_ppp = pd.DataFrame(index=events_index_ppp, columns=events_col_ppp)

for i in events_col_ppp:
    data_events_ppp[i] = np.where((data_ppp[i] - data_ppp['Mkt']) > daily_diff, 1, np.where((data_ppp[i] - data_ppp['Mkt']) < -daily_diff, -1, np.nan))

print("Evaluation Complete")

L1 = 30
window = 20

pos_dict_ppp = defaultdict(dict)
neg_dict_ppp = defaultdict(dict)

for s in tqdm(events_col_ppp):
    pos_event_dates_ppp = data_events_ppp[s][data_events_ppp[s] == 1].index.tolist()
    neg_event_dates_ppp = data_events_ppp[s][data_events_ppp[s] == -1].index.tolist()

    pos_dict_s_ppp = defaultdict(dict)
    neg_dict_s_ppp = defaultdict(dict)

    for pos_event in pos_event_dates_ppp:
        date_loc_ppp = data_ppp.index.get_loc(pos_event)
        date_loc_ppp = date_loc_ppp - window

        if date_loc_ppp > L1 and date_loc_ppp <= len(data_ppp) - (2*window+1):
            index_range = (2*window) + 1

            pos_dict_s_event_ppp = OrderedDict()
            for d in range(index_range):
                date_loc_ppp2 = date_loc_ppp + d

                u_i = data_ppp[s][date_loc_ppp2 - L1: date_loc_ppp2 - 1].mean()
                u_m = data_ppp['Mkt'][date_loc_ppp2 - L1: date_loc_ppp2 - 1].mean()
                R_i = data_ppp.iloc[date_loc_ppp2, data_ppp.columns.get_loc(s)]
                R_m = data_ppp.iloc[date_loc_ppp2, data_ppp.columns.get_loc('Mkt')]
                beta_i = ((R_i - u_i) * (R_m - u_m)) / (R_m - u_m) ** 2
                alpha_i = u_i - (beta_i * u_m)
                var_err = (1 / (L1 - 2)) * (R_i - alpha_i - (beta_i * R_m)) ** 2
                AR_i = R_i - alpha_i - (beta_i * R_m)

                pos_dict_s_event_ppp[date_loc_ppp2] = AR_i

            pos_dict_s_ppp[pos_event] = pos_dict_s_event_ppp

    pos_dict_ppp[s] = pos_dict_s_ppp


    for neg_event in neg_event_dates_ppp:
        date_loc_ppp = data_ppp.index.get_loc(neg_event)
        date_loc_ppp = date_loc_ppp - window

        if date_loc_ppp > L1 and date_loc_ppp <= len(data_ppp) - (2*window+1):
            index_range = (2*window) + 1

            neg_dict_s_event_ppp = OrderedDict()
            for d in range(index_range):
                date_loc_ppp2 = date_loc_ppp + d

                u_i = data_ppp[s][date_loc_ppp2 - L1: date_loc_ppp2 - 1].mean()
                u_m = data_ppp['Mkt'][date_loc_ppp2 - L1: date_loc_ppp2 - 1].mean()
                R_i = data_ppp.iloc[date_loc_ppp2, data_ppp.columns.get_loc(s)]
                R_m = data_ppp.iloc[date_loc_ppp2, data_ppp.columns.get_loc('Mkt')]
                beta_i = ((R_i - u_i) * (R_m - u_m)) / (R_m - u_m) ** 2
                alpha_i = u_i - (beta_i * u_m)
                var_err = (1 / (L1 - 2)) * (R_i - alpha_i - (beta_i * R_m)) ** 2
                AR_i = R_i - alpha_i - (beta_i * R_m)

                neg_dict_s_event_ppp[date_loc_ppp2] = AR_i

            neg_dict_s_ppp[neg_event] = neg_dict_s_event_ppp

    neg_dict_ppp[s] = neg_dict_s_ppp



abret_col_ppp = ready_permnos_ppp[:]
abret_col_ppp.remove("Mkt")
abret_ppp_index = range(-window, window + 1)
pos_data_abret_ppp = pd.DataFrame(index=abret_ppp_index, columns=abret_col_ppp)
neg_data_abret_ppp = pd.DataFrame(index=abret_ppp_index, columns=abret_col_ppp)

for h in abret_col_ppp:
    if h in pos_dict_ppp.keys():
        for z in abret_ppp_index:

            pos_data_abret_ppp[h][z] = np.mean([list(x.values())[z + window] for x in pos_dict_ppp[h].values()])

for f in abret_col_ppp:
    if f in neg_dict_ppp.keys():
        for v in abret_ppp_index:
            neg_data_abret_ppp[f][v] = np.mean([list(x.values())[v + window] for x in neg_dict_ppp[f].values()])

print("Done Setting Up for Plots")

pos_CAR_ppp = pos_data_abret_ppp.cumsum()
neg_CAR_ppp = neg_data_abret_ppp.cumsum()

pos_CAR_ppp['SUM'] = pos_CAR_ppp.sum(axis=1)
neg_CAR_ppp['SUM'] = neg_CAR_ppp.sum(axis=1)

plt.clf()
plt.plot(pos_CAR['SUM'], color = "blue", label = "Non-PPP Loan Receivers")
plt.plot(pos_CAR_ppp['SUM'], color = "red", label = "PPP Loan Receivers")
plt.legend()
plt.ylabel('CAR')
plt.title("Positive Abnormal Return (2% Event)")
plt.xlabel('Window')
matplotlib.rcParams.update({'font.size': 8})
plt.savefig('PositiveCAR_2Percent.png', format='png')

plt.clf()
plt.plot(neg_CAR['SUM'], color = "blue", label = "Non-PPP Loan Receivers")
plt.plot(neg_CAR_ppp['SUM'], color = "red", label = "PPP Loan Receivers")
plt.legend()
plt.ylabel('CAR')
plt.title("Negative Abnormal Return (2% Event)")
plt.xlabel('Window')
matplotlib.rcParams.update({'font.size': 8})
plt.savefig('NegativeCAR_2Percent.png', format='png')


stop_time = timeit.default_timer()

print("Time: ", stop_time - start_time)
