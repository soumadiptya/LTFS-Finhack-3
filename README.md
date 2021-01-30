# LTFS Fin Hack 3
<img src="logo.png" alt="Logo">

## Problem Statement

As a Financial Institution LTFS wants to predict which customers will avail Top Up loans on their existing loans and at what point during their Loan Tenures will they do so. This is a Multi Class classification Problem with the following available class labels:-  
* 12-18 Months
* 18-24 Months
* 24-30 Months
* 30-36 Months
* 36-48 Months
* '>48 Months  

Following Datasets are available for the task:- 

1.**Customer’s Demographics**: The demography table along with the target variable & demographic information contains variables related to Frequency of the loan, Tenure of the loan, Disbursal Amount for a loan & LTV.  

2.**Bureau data**:  Bureau data contains the behavioural and transactional attributes of the customers like current balance, Loan Amount, Overdue etc. for various tradelines of a given customer  

Based on intuition a combination of Demographic and Loan features are likely to be the most important predictors for the task such as Current Amount remaining, Customers total Loan Amounts outstanding, Customer Income etc. 
