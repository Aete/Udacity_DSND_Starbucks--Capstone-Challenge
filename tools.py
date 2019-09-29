import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns


def id_mapper(df,id_column):
    '''
    This function is for mapping user's id to int.
    --------
    Input:

    df: the pandas dataframe which contain user id column.    
    id_column: title(str) of user id column
    --------
    Output:
    
    coded_dict: dict with keys as user ids, and items as int numbers
    
    '''
    coded_dict = dict()
    cter = 1
    
    for val in df[id_column]:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
    return coded_dict


# make a function for all of processes for cleaning the profile dataset.
def profile_cleaning(df):
    '''
    This function is for cleaning the profile dataset
    --------
    Input:

    df: the profile dataset (the pandas dataframe)    
    --------
    Output:
    
    df: cleaned profile dataset
    peope_remove: a list(pandas Series) of people who have missing values. 
    
    '''      
    # move the 'id' column to the first of the dataset
    df = df[df.columns.tolist()[3:] + df.columns.tolist()[:3]] 
    
    # change id to int
    user_dict = id_mapper(df,'id')
    df['id']=df['id'].apply(lambda x: user_dict[x])
    
    # change genter to int
    gen_dict = dict()
    keys = df['gender'].unique().tolist()
    for i in range(len(keys)):
        gen_dict[keys[i]] = i
    df['gender'] = df['gender'].replace(gen_dict)
    
    # change became_member_on data to numbers of days from the signing date.
    today_ = pd.to_datetime('today')
    df['became_member_on'] = (today_- pd.to_datetime(df['became_member_on'],format='%Y%m%d')).astype('timedelta64[D]').astype(int)
    
    # remove rows with missing values
    missing_indice = df[df['age']==118].index
    people_remove = df.loc[missing_indice]['id']
    df=df.drop(missing_indice,axis=0)
    
    return df, people_remove, user_dict


# Below code is for change lists in channaels and offer_type comlumn to one-hot encoded data of each categories.
def one_hot_encoding(df, target_column, categories):
    '''
    This is for changing categorical column to several columns of one_hot encoded.
    
    Input:
            df: Pandas DataFrame
            target_column: name of categorical column which we have to change
            categories: List of categories
            
    Output:
            df_new: Pandas DataFrame

    '''
    
    for i in categories:
        df[i] = df[target_column].apply(lambda x: 1 if i in x else 0)
    
    df_new = df.drop(target_column,axis=1)
    
    return df_new

# make a function for all of processes for cleaning the portfolio dataset.
def portfolio_cleaning(df):
    
    '''
    This function is for cleaning the profile dataset
    --------
    Input:

    df: the profile dataset (the pandas dataframe)    
    --------
    Output:
    
    df: cleaned portfolio dataset
    offer_dict: dict for changing offer_id to simple int id
    
    '''  
    # change id to int
    offer_id_dict = id_mapper(df,'id')
    df['id'] = df['id'].apply(lambda x: offer_id_dict[x])
    
    # one hot encoding for categorical values
    df = one_hot_encoding(df,'offer_type',['bogo','informational','discount'])
    df = one_hot_encoding(df,'channels',['email','mobile','web','social'])
    
    # change value from days to hours
    df['duration'] = df['duration']*24
    
    # change column order
    columns = df.columns.tolist()
    columns = columns[2::-1] + columns[3:]
    df = df[columns]
    
    # drop the 'email' column because it is same for all of the rows
    df = df.drop('email',axis=1)
    
    return df, offer_id_dict


def viewed_check(x,offer_viewed):
    '''
    This function is for judging the specific offer was viewed by specific customer. 
    It will return 1 if the customer viewed the offer. It will return 0 if the customer did not viewed the offer.
    --------
    Input:

    x: the offer_recieved dataset or the subset of offer_recieved (the pandas dataframe)    
    --------
    output:
    
    0: the customer did not viewed the offer.
    1: the customer viewed the offer.
    
    '''  
    # create person, offer, duration variables which belongs to a specific offer
    person = x['person']
    offer = x['offer_id']
    duration = x['duration']
    time = x['time']
    
    # make a list of time, when offers were viewed, from 'offer_viewed' dataset based on person and offer information
    viewed_list = offer_viewed.loc[(offer_viewed['person']==person) & (offer_viewed['offer_id']==offer)]['time'].tolist()
    for v in viewed_list:
        if ((v-time)<duration) & ((v-time)>=0):
            return 1
    return 0

def completed_check(x,offer_completed):
    '''
    This function is for judging the specific offer was completed by specific customer. 
    It will return 1 if the customer completed the offer by duration. It will return 0 if the customer did not completed the offer.
    --------
    Input:

    x: the offer_recieved dataset or the subset of offer_recieved (the pandas dataframe)    
    --------
    output:
    
    0: the customer did not completed the offer.
    1: the customer completed the offer.
    
    '''  
    # create person, offer, duration variables which belongs to a specific offer
    person = x['person']
    offer = x['offer_id']
    duration = x['duration']
    time = x['time']
    
    # make a list of time, when offers were completed, from 'offer_completed' dataset based on person and offer information
    completed_list = offer_completed.loc[(offer_completed['person']==person) & (offer_completed['offer_id']==offer)]['time'].tolist()
    
    # judging the specific offer was completed by specific customer before duration was overed
    for c in completed_list:
        if ((c-time)<duration) & ((c-time)>=0):
            return 1
    return 0

def past_completed_count(x, offer):
    '''
    This function is for counting previous completed numbers
    
    Input:
        x: the offer_recieved dataset or the subset of offer_recieved (the pandas dataframe)     
        offer: 'offer_received' dataframe from 'completed_check' function
    ------
    Output:
        count: the number of offers which were completed in the past.
    '''
    
    time = x['time']
    person = x['person']
    offer_list = offer.loc[(offer['time']<time) & (offer['person']==person)]
    count = 0
    if len(offer_list)>0:
        count = offer_list['completed'].sum()
    return count

def offer_processing(df,profile, portfolio,offer_id_dict):
        
    '''
    This function is for pre-processing the 'offer' dataframe from the transcript dataset
    --------
    Input:

    df: the 'offer' dataframe from 'transcript' dataframe 
    profile: the cleaned profile dataset
    portfolio: the cleaned portfolio dataset
    offer_id_dict: A dictionary with with 'offer id' as its keys and int values as its items
    --------
    Output:
    
    offer_received: the processed 'offer' dataframe
    
    '''   
    
    # extract 'offer_id' from 'value' column
    df['offer_id'] = df['value'].apply(lambda x:list(x.values())[0])

    # map 'offer_id' column with the dictionary which we made before
    df['offer_id'] = df['offer_id'].apply(lambda x: offer_id_dict[x])

    # drop the 'value' column
    df = df.drop(['value'],axis=1)
    
    # merge the dataset with the 'profile' and 'portfolio' dataset
    df = df.merge(profile, left_on='person', right_on='id')
    df = df.merge(portfolio, left_on='offer_id', right_on = 'id')
    df = df.drop(['id_x','id_y'],axis=1)
    offer_received = df.loc[df['event']=='offer received']
    offer_viewed = df.loc[df['event']=='offer viewed']
    offer_completed = df.loc[df['event']=='offer completed']    
    offer_received['viewed'] = 0
    offer_received['completed'] = 0
    offer_received['viewed'] = offer_received[['person','offer_id','time','duration']].apply(lambda x: viewed_check(x,offer_viewed),axis=1)    
    offer_received['completed'] = offer_received.apply(lambda x: completed_check(x,offer_completed),axis=1)
    offer_received['completed_count'] = offer_received.apply(lambda x: past_completed_count(x,offer_received), axis=1)
    offer_received = offer_received.drop('event',axis=1)
    return offer_received

def transcript_cleaning(df, profile, portfolio, user_dict, offer_id_dict, people_remove):
    
    '''
    This function is for cleaning the transcript dataset
    --------
    Input:

    df: the transcript dataset (the pandas dataframe) 
    profile: the cleaned profile dataset
    portfolio: the cleaned portfolio dataset
    user_dict: A dictionary with 'user id' as its keys and int values as its items
    offer_id_dict: A dictionary with with 'offer id' as its keys and int values as its items
    --------
    Output:
    
    df: cleaned portfolio dataset
    offer_dict: dict for changing offer_id to simple int id
    
    '''  
    
    # mapping person's id with the dictionary which we made before
    df['person']=df['person'].apply(lambda x: user_dict[x])

    # this part is for removing raws for people who are identified as missing data
    missing_idx = df[df['person'].isin(people_remove)].index
    df=df.drop(missing_idx,axis=0)
    offer = df[df['event'] != 'transaction']
    transaction = df[df['event'] == 'transaction']
    
    # this part is for cleaning offer part
    offer = offer_processing(offer,profile, portfolio,offer_id_dict)
    
    # this part is for cleaning transaction part
    transaction['amount'] = transaction['value'].apply(lambda x:x['amount'])
    transaction = transaction.drop('value',axis=1)
    transaction = transaction.merge(profile, left_on='person', right_on='id')
    return offer, transaction