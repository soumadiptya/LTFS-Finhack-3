# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:53:57 2021
This file contains functions to generate features for Bureau data
@author: Soumadiptya.c
"""
#%% Import libraries
import numpy as np
import pandas as pd
import re

#%% Helper Functions
def amount_correction(amount):
    '''Used to correct numbers stored as strings'''
    amount = re.sub(",", "", str(amount))
    amount = int(amount)
    amount = amount * -1 if amount < 0 else amount
    return amount

def correct_reported_dt(reported_dt, reported_dt_hist, last_payment_dt):
    '''Used to correct missing Reported Dates. If Reported dt is missing replace with
    newest date from Reported dt hist. If that too is missing replace with last
    payment date'''
    if str(reported_dt) == "NaT":
        try:
            extracted_dt = re.findall('[0-9]+', reported_dt_hist)[0]
            reported_dt = pd.to_datetime(extracted_dt)
        except:
            reported_dt = last_payment_dt
    return reported_dt

def correct_disbursed_dt(disbursed_dt, reported_dt_hist, reported_dt):
    '''Used to correct missing Disbursed Dates. If Disbursed dt is missing replace with
    oldest date from Reported dt hist. If that too is missing replace with last
    reported date'''
    if str(disbursed_dt) == "NaT":
        try:
            extracted_dt = re.findall('[0-9]+', reported_dt_hist)[-1]
            disbursed_dt = pd.to_datetime(extracted_dt)
        except:
            disbursed_dt = reported_dt
    return disbursed_dt

def treat_overdue_amt(overdue_amt, overdue_amt_hist):
    '''Used to correct Overdue Amounts'''
    if str(overdue_amt) == 'nan':
        try:
            overdue_amt = re.findall('^[0-9]+', overdue_amt_hist)[0]
        except:
            overdue_amt = 0
    return overdue_amt

def correct_close_dt(close_dt, reported_dt):
    '''Correct and clean clean close dates that are not possible'''
    if close_dt is not None:
        if '2999' in str(close_dt) or '3000' in str(close_dt) or '2802' in str(close_dt):
            close_dt = reported_dt
    return close_dt

def convert_scientific_notation_to_string(scientific_notation_string):
    '''Due to storage Issues some DPD history are stored as exponential numbers.
    This function is used to get proper strings from such cases'''
    # This function works correctly as long as number after E>1
    if 'E' in scientific_notation_string:
        power = re.findall('[0-9]+$', scientific_notation_string)[0]
        base_num = re.findall('^[0-9]+\.{0,}[0-9]*', scientific_notation_string)[0]
        base_num = re.sub('\.', '', base_num)
        actual_num = base_num + '0'*(int(power)-2)
    else:
        actual_num = scientific_notation_string
    return actual_num

def get_dpd_list(dpd_hist, len_reported_date_hist):
    '''This function is used to get proper list of DPD and pad or shrink as needed'''
    dpd_list = re.findall('...', dpd_hist)
    # Pad list to length of Reported date Hist
    if len(dpd_list) < len_reported_date_hist:
        padding = len_reported_date_hist - len(dpd_list)
        dpd_list = ['0']*padding + dpd_list
    elif len(dpd_list) > len_reported_date_hist:
        dpd_list = dpd_list[:len_reported_date_hist]
    return dpd_list

def equalize_list_lengths(orig_list, len_reported_date_hist):
    if len(orig_list) < len_reported_date_hist:
        padding = len_reported_date_hist - len(orig_list)
        orig_list = [0] * padding + orig_list
    elif len(orig_list) > len_reported_date_hist:
        orig_list = orig_list[:len_reported_date_hist]
    return orig_list

def create_previous_loan_history(previous_loans_df):
    '''This function is used to create a historical Dataframe containing various
    details of earlier loans in long format'''
    previous_loans_df_expanded = pd.DataFrame(columns=['Reported Date', 'Cur Bal', 'Overdue Amt', 'DPD'])
    for i in np.arange(previous_loans_df.shape[0]):
        cur_bal_hist = pd.Series(previous_loans_df.loc[i, 'CUR BAL - HIST'].split(','))
        cur_bal_hist = cur_bal_hist[:-1]
        overdue_amt_hist = pd.Series(previous_loans_df.loc[i, 'AMT OVERDUE - HIST'].split(','))
        overdue_amt_hist = overdue_amt_hist[:-1]
        reported_dt_hist = pd.Series(previous_loans_df.loc[i, 'REPORTED DATE - HIST'].split(','))
        reported_dt_hist = reported_dt_hist[:-1]
        # Equalize list lengths 
        cur_bal_hist = equalize_list_lengths(list(cur_bal_hist), len(reported_dt_hist))
        overdue_amt_hist = equalize_list_lengths(list(overdue_amt_hist), len(reported_dt_hist))
        dpd_hist = get_dpd_list(convert_scientific_notation_to_string(previous_loans_df.loc[i, 'DPD - HIST']), len(reported_dt_hist))
        cur_bal_df = pd.DataFrame({'Reported Date':reported_dt_hist, 'Cur Bal':cur_bal_hist,
                                   'Overdue Amt':overdue_amt_hist, 'DPD':dpd_hist})
        cur_bal_df['Previous Loan Index'] = i
        previous_loans_df_expanded = pd.concat([previous_loans_df_expanded, cur_bal_df], axis=0)
    # Fill blank Cur Bal and Overdue Amt with 0 and convert to number
    previous_loans_df_expanded['Cur Bal'] = np.where(previous_loans_df_expanded['Cur Bal'] == "",
                                                     0, previous_loans_df_expanded['Cur Bal'])
    previous_loans_df_expanded['Overdue Amt'] = np.where(((previous_loans_df_expanded['Overdue Amt'] == "")|(previous_loans_df_expanded['Overdue Amt'].isna())),
                                                     0, previous_loans_df_expanded['Overdue Amt'])
    previous_loans_df_expanded['Cur Bal'] = previous_loans_df_expanded['Cur Bal'].astype(int)
    previous_loans_df_expanded['Overdue Amt'] = previous_loans_df_expanded['Overdue Amt'].astype(int)
    # Convert DPD to numbers properly 
    previous_loans_df_expanded['DPD'] = previous_loans_df_expanded['DPD'].map(lambda x:re.sub('DDD|XXX', '0', x))
    previous_loans_df_expanded['DPD'] = previous_loans_df_expanded['DPD'].astype(int)
    # Convert Reported Date to datetime
    previous_loans_df_expanded['Reported Date'] = pd.to_datetime(previous_loans_df_expanded['Reported Date'])
    previous_loans_df_expanded['Previous Loan Index'] = previous_loans_df_expanded['Previous Loan Index'].astype(int)
    previous_loans_df_expanded = previous_loans_df_expanded.loc[:, ['Previous Loan Index', 'Reported Date', 'Cur Bal', 'Overdue Amt', 'DPD']]
    previous_loans_df_expanded.reset_index(drop=True, inplace=True)
    return previous_loans_df_expanded

# Now create a function which uses the current Loan Index and the History Data Frame to get the historical features
def get_historical_loan_feats(previous_loans_df_expanded, current_disbursal_dt):
    ''' This is the actual function to generate the historical features for a customers
    previous loans'''
    previous_loans_df_expanded['Current Loan Disbursal Date'] = current_disbursal_dt
    previous_loans_df_expanded['Difference Disbursal Reported dt'] = \
    np.abs(previous_loans_df_expanded['Current Loan Disbursal Date'] - previous_loans_df_expanded['Reported Date'])
    previous_loans_df_expanded['Min difference per Account'] = \
    previous_loans_df_expanded.groupby('Previous Loan Index')['Difference Disbursal Reported dt'].transform('min')
    count_overdue_payments = previous_loans_df_expanded[previous_loans_df_expanded['Overdue Amt'] > 0].shape[0] # Overdue Amt > 0
    count_payments_past_due = previous_loans_df_expanded[previous_loans_df_expanded['DPD'] > 0].shape[0]
    max_overdue_payment = previous_loans_df_expanded['Overdue Amt'].max()
    max_dpd = previous_loans_df_expanded['DPD'].max()
    previous_loans_df_expanded = \
    previous_loans_df_expanded[previous_loans_df_expanded['Difference Disbursal Reported dt'] == previous_loans_df_expanded['Min difference per Account']]
    total_outstanding_bal = previous_loans_df_expanded['Cur Bal'].sum()
    total_overdue_amt = previous_loans_df_expanded['Overdue Amt'].sum()
    total_dpd = previous_loans_df_expanded['DPD'].sum()
    historical_feats = [total_outstanding_bal, total_overdue_amt, count_overdue_payments, max_overdue_payment, total_dpd, count_payments_past_due, max_dpd]
    return historical_feats

def generate_historical_loan_feats_customer_level(customer_df):
    ''' This function generates the Historical features for each Loan Account and
    returns them as a Dataframe. Calls the previous functions as needed'''
    customer_df['Cur_Bal_Prev_Loans'] = 0
    customer_df['Cur_Overdue_Bal_Prev_Loans'] = 0
    customer_df['Count_Overdue_Payments'] = 0
    customer_df['Max_Overdue_Amount'] = 0
    customer_df['Total DPD'] = 0
    customer_df['Count Past Due Payments'] = 0
    customer_df['Max DPD'] = 0
    # 1) No of past Loan accounts
    customer_df.sort_values('DISBURSED-DT', inplace=True)
    customer_df.reset_index(drop=True, inplace=True)
    customer_df['Previous Loan Accounts'] = customer_df.index
    # 2) No of Active Loan accounts at the time of disbursal
    customer_df['Previous_Active_Disbursal_DT'] = 0
    customer_df['Same_Loan_Type_count'] = 0
    customer_df['Unique_Acct_Type_count'] = 0
    for i in np.arange(customer_df.shape[0]):
        disbursal_dt = customer_df.loc[i, 'DISBURSED-DT']
        last_active_dt = customer_df.loc[i, 'Last Active Date']
        loan_type_current = customer_df.loc[i, 'ACCT-TYPE']
        # Create the History Data of previous loans
        if i!=customer_df.shape[0]-1:
            temp_data_1 = customer_df.loc[:i, :]
            temp_data_1 = temp_data_1[temp_data_1['Last Active Date'] > disbursal_dt]
            temp_data_1 = temp_data_1.iloc[1:, :]
            temp_data_1.reset_index(drop=True, inplace=True)
            temp_data_2 = customer_df.loc[i+1:, :]
            temp_data_2 = temp_data_2[temp_data_2['DISBURSED-DT'] < last_active_dt]
            temp_data_2.reset_index(drop=True, inplace=True)
            temp_data = pd.concat([temp_data_1, temp_data_2], axis=0)
            temp_data.reset_index(drop=True, inplace=True)
            active_loan_act_count = temp_data.shape[0]
            customer_df.loc[i, 'Previous_Active_Disbursal_DT'] = active_loan_act_count
            # Any other Active Loans of the same type
            same_loan_type_df = temp_data[temp_data['ACCT-TYPE'] == loan_type_current]
            same_loan_type_count = same_loan_type_df.shape[0]
            customer_df.loc[i, 'Same_Loan_Type_count'] = same_loan_type_count
            # Unique Acct types count
            unique_acct_types_count = temp_data['ACCT-TYPE'].nunique() + 1
            customer_df.loc[i, 'Unique_Acct_Type_count'] = unique_acct_types_count
        else:
            temp_data = customer_df.loc[:i, :]
            temp_data = temp_data[temp_data['Last Active Date'] > disbursal_dt]
            temp_data.reset_index(drop=True, inplace=True)
            active_loan_act_count = temp_data.shape[0]
            customer_df.loc[i, 'Previous_Active_Disbursal_DT'] = active_loan_act_count - 1
            # Any other Active Loans of the same type
            same_loan_type_df = temp_data[temp_data['ACCT-TYPE'] == loan_type_current]
            same_loan_type_count = same_loan_type_df.shape[0]
            customer_df.loc[i, 'Same_Loan_Type_count'] = same_loan_type_count
            # Unique Acct types count
            unique_acct_types_count = temp_data['ACCT-TYPE'].nunique()
            customer_df.loc[i, 'Unique_Acct_Type_count'] = unique_acct_types_count
        # Expand the History Data Frames
        # No previous active loans may exist
        try:
            temp_data_expanded = create_previous_loan_history(temp_data.loc[:, ['ID', 'CUR BAL - HIST', 'REPORTED DATE - HIST', 'AMT OVERDUE - HIST', 'DPD - HIST']])
            customer_df.loc[i, 'Cur_Bal_Prev_Loans'] = get_historical_loan_feats(temp_data_expanded, disbursal_dt)[0]
            customer_df.loc[i, 'Cur_Overdue_Bal_Prev_Loans'] = get_historical_loan_feats(temp_data_expanded, disbursal_dt)[1]
            customer_df.loc[i, 'Count_Overdue_Payments'] = get_historical_loan_feats(temp_data_expanded, disbursal_dt)[2]
            customer_df.loc[i, 'Max_Overdue_Amount'] = get_historical_loan_feats(temp_data_expanded, disbursal_dt)[3]
            customer_df.loc[i, 'Total DPD'] = get_historical_loan_feats(temp_data_expanded, disbursal_dt)[4]
            customer_df.loc[i, 'Count Past Due Payments'] = get_historical_loan_feats(temp_data_expanded, disbursal_dt)[5]
            customer_df.loc[i, 'Max DPD'] = get_historical_loan_feats(temp_data_expanded, disbursal_dt)[6]
        except:
            customer_df.loc[i, 'Cur_Bal_Prev_Loans'] = -1
            customer_df.loc[i, 'Cur_Overdue_Bal_Prev_Loans'] = -1
            customer_df.loc[i, 'Count_Overdue_Payments'] = -1
            customer_df.loc[i, 'Max_Overdue_Amount'] = -1
            customer_df.loc[i, 'Total DPD'] = -1
            customer_df.loc[i, 'Count Past Due Payments'] = -1
            customer_df.loc[i, 'Max DPD'] = -1
    return customer_df
