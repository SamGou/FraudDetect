# Data
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform

# Graphing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import ConfusionMatrixDisplay


# I/O
import os
from tqdm import tqdm
from pprint import pprint
import argparse


# ML
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn import metrics
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

ALL_TRANSACTIONS = "data/transactions_obf.csv"
FRAUD_IDS = "data/labels_obf.csv"
AUGMENTED_TRANSACTIONS = "data/transactions_aug_obf.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true",help="Plot EDA graphs")
parser.add_argument("--analysis",action="store_true",help="Print EDA analysis")
parser.add_argument("--seed",default=2041999,type=int,help="Define the seed. Default seed is 2041999 and if changed it is recommended that the random searches are performed again with --rand_search.")
parser.add_argument("--historical", action="store_true",help="Add historical merchant information from the training set as a trustworthiness score feature to both training and test data")
parser.add_argument("--regen", action="store_true",help="Recreate the initial pre-processed dataset for faster loading. Useful if 'load_dataset' function has been changed")
parser.add_argument("--rand-search",action="store_true",help="Perform random search with XGClassifier")
parser.add_argument("--sampling",default="smote_oversample",help="Choose sampling technique. Options are: [smote_oversample, near_miss_undersample, random_undersample]")
args = parser.parse_args()

PLOT = args.plot # Plot basic statistics about the data
PRINT_ANALYSIS = args.analysis # Print analysis and statistics
SEED = args.seed
HISTORICAL = args.historical

def load_dataset(regen,from_save=True):
    """Function which loads the dataset from disk for faster processing. If the dataset doesnt exist it will generate a new one.

    Args:
        regen (bool): Recreate the dataset and save it
        from_save (bool, optional): Load data from a saved file. Defaults to True.
    """
    # Load data from saved file (if it exists) for faster processing
    if os.path.exists(AUGMENTED_TRANSACTIONS) and from_save and not regen:
        print("Loading processed data from disk")
        return pd.read_csv(AUGMENTED_TRANSACTIONS)
    else:
        print("Processing and saving new dataset...")
        # Load data and fraud labels
        all_transactions = pd.read_csv(ALL_TRANSACTIONS)
        fraudLabels = pd.read_csv(FRAUD_IDS)[["eventId"]]
        
        # Create a new column called "class"
        all_transactions["class"] = 0
        fraudLabels["class"] = 1
        
        # Merge DataFrames on common column
        temp_merged = pd.merge(all_transactions,fraudLabels,on="eventId",how="left")
        
        # Replace values based on non-null condition
        temp_merged.loc[~temp_merged["class_y"].isnull(),'class_x'] = temp_merged["class_y"]
        
        # Drop the new column and change the name of the old one
        temp_merged = temp_merged.drop(["class_y"],axis=1)
        all_transactions = temp_merged.rename(columns={"class_x":"class"})
        del temp_merged
        
        # TIME CONVERSION 
        # the time feature as it is currently can be better represented as a month, day and hour
        # these representations will be used during EDA and for training they will be converted to cyclical features
        
        # EXTRACTING TIME FEATURES
        def _extract_time_features(string):
            """Mapping function which extracts the time features from the "transactionTime" string"""
            
            timestamp = datetime.strptime(string["transactionTime"],"%Y-%m-%dT%H:%M:%SZ")
            return pd.Series({"year": timestamp.year,
                            "month": timestamp.month,
                            "day": timestamp.day,
                            "hour": timestamp.hour})
        
        def _convert_time_features(string):
            """Mapping function which transforms time into a cyclical feature"""
            
            timestamp = datetime.strptime(string["transactionTime"],"%Y-%m-%dT%H:%M:%SZ")
            
            # Convert time to a cyclical feature to make better use of the information encoded during tranining 
            total_seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
            norm_time = total_seconds / (24*3600)
            sin_time,cos_time = (np.sin(2*np.pi*norm_time),
                                np.cos(2*np.pi*norm_time))
            
            # Convert date to a cyclical feature
            start_date = datetime(2017,1,1) # The first date from which counting starts
            total_days = (timestamp-start_date).days + 1 # adding one includes the current day in the calculation
            norm_date = total_days / 365
            sin_date,cos_date = (np.sin(2*np.pi*norm_date),
                                np.cos(2*np.pi*norm_date))
    
            return pd.Series({"sin_time": sin_time,
                            "cos_time": cos_time,
                            "sin_date": sin_date,
                            "cos_date": cos_date})
        
        # Map time values
        new_cols = all_transactions.apply(_extract_time_features,axis=1)
        all_transactions[["year","month","day","hour"]] = new_cols
        aug_time = all_transactions.apply(_convert_time_features,axis=1)
        all_transactions[["sin_time","cos_time","sin_date","cos_date"]] = aug_time
        
        # Drop transactionTime feature and keep only 2017 examples
        all_transactions= all_transactions.drop(["transactionTime"],axis=1)
        all_transactions = all_transactions.loc[all_transactions["year"]==2017]
        
        #Save the data for faster loading
        if not os.path.exists(AUGMENTED_TRANSACTIONS) or regen:
            all_transactions.to_csv(AUGMENTED_TRANSACTIONS)        
        return all_transactions

# Data processing
def pos_mapping(x):
    """Mapping function for the posEntryMode feature. Each category is mapped to from 0 to 8"""
    
    # The "79" category is not present in the dictionary file and will therefore be bucketed with "0" as an unknown value
    key = {0:0,
           79:0,
           1:1,
           2:2,
           5:3,
           7:4,
           80:5,
           81:6,
           90:7,
           91:8}
    return key[x["posEntryMode"]]

def keepOnlyTop5Countries(x,top5,key):
        '''A function which maps the top 5 most common countries to an integer
        and every other, rarer country is categorised as a separate integer'''
        
        if x["merchantCountry"] in top5:
            return key[x["merchantCountry"]]
        else:
            return 5

def map_merchantZip(x):
    """Mapping function which categorizes 'merchantZip' as one of 3 categories - 1 (missing value) and 0 (unknown value) as unknowns and 2 denoting a postcode was present"""
    if x["merchantZip"]==1 or x["merchantZip"]=="0":
        return int(x["merchantZip"])
    else:
        return 2

def one_hot_encode_features(df):
    """Function which applies all one-hot encoding pre-processing steps to the features.
    Args:
        df (pd.DataFrame): DataFrame with features to be one-hot encoded

    Returns:
        df (pd.DataFrame): One-hot encoded and pre-processed dataset
    """
    # Augment "merchantCountry" to combine less common categories into one category, and one hot encode 
    top5 = df["merchantCountry"].value_counts().head(5).index
    key = {i:new_ind for new_ind,i in enumerate(top5)}
    dataAugmented = df.apply(keepOnlyTop5Countries,args=(top5,key,),axis=1)
    df["merchantCountry"]= dataAugmented
    one_hot_encoded = pd.get_dummies(df['merchantCountry'], prefix='mercCountry')
    df = pd.concat([df, one_hot_encoded.map(int)], axis=1)
    
    # Apply mapping function to posEntryMode and one hot encode for use in the training
    dataAugmented = df.apply(pos_mapping,axis=1)
    df["posEntryMode"]= dataAugmented
    one_hot_encoded = pd.get_dummies(df['posEntryMode'], prefix='pos')
    df = pd.concat([df, one_hot_encoded.map(int)], axis=1)
    
    # Fill null values with a dummy variable, combine and one hot encode merchantZip
    df['merchantZip'] = df['merchantZip'].fillna(value=1)
    df["merchantZip"] = df.apply(map_merchantZip,axis=1)
    one_hot_encoded = pd.get_dummies(df['merchantZip'], prefix='merczip')
    df = pd.concat([df, one_hot_encoded.map(int)], axis=1)

    # One hot encode availableCash and transactionAmount
    # Bin transaction amount and availableCash to capture information about amount instead of spread
    bins = [0,2000,4000,6000,8000,10000,12000,float("inf")]
    df["transactionAmount_binned"] = pd.cut(df["transactionAmount"],bins=bins,labels=False,right=False)
    one_hot_encoded = pd.get_dummies(df['transactionAmount_binned'], prefix='transamount_bin')
    df = pd.concat([df, one_hot_encoded.map(int)], axis=1)
    
    df["availableCash_binned"] = pd.cut(df["availableCash"],bins=bins, labels=False, right=False)
    one_hot_encoded = pd.get_dummies(df['availableCash_binned'], prefix='availcash_bin')
    df = pd.concat([df, one_hot_encoded.map(int)], axis=1)
    
    return df

def remove_outliers(df,iqr_mult=1.5):
    """Function which removes the outliers from the "transactionAmount" feature, where outliers are defined as values outside a lower and upper bound 

    Args:
        df (pd.DataFrame): _description_
        iqr_mult (float, optional): Multiplier which defines how sensitive the outlier detection is. Defaults to 1.5.

    Returns:
        outliers_removed (pd.DataFrame): DataFrame with all outliers removed
        outliers_only (pd.DataFrame): DataFrame containing only the outliers 
    """
    # Find the transactionAmount IQR and find outlier bounds
    q25,q75 = np.percentile(df["transactionAmount"],25), np.percentile(df["transactionAmount"],75)
    iqr = q75-q25
    lower = q25 - iqr*iqr_mult
    upper = q75 + iqr*iqr_mult
    
    # Remove outliers and separate into two dataframes normal data and outliers only
    outliers_removed = df.loc[(df["transactionAmount"] >= lower) & (df["transactionAmount"] <= upper)]
    print("OUTLIERS REMOVED: ", df["transactionAmount"].count()-outliers_removed["transactionAmount"].count())
    
    outliers_only = df.loc[(df["transactionAmount"] < lower) | (df["transactionAmount"] > upper)]
    return outliers_removed,outliers_only

def stratified_split(X,y):
    """Function which splits a dataset into two parts of predefined sizes (90:10) using a stratification strategy.

    Args:
        X (pd.DataFrame): DataFrame of the features and observations
        y (pd.DataFrame): DataFrame of the target label

    Returns:
        original_Xtrain(pd.DataFrame): DataFrame of the features and observations
        original_ytrain(pd.DataFrame): DataFrame of the target label
        original_Xtest(pd.DataFrame): DataFrame of the test set with target labels
    """
    # Split data into 90:10 train:test ratio using stratified split to keep class distribution the same
    strat = StratifiedShuffleSplit(n_splits = 1, train_size=0.9, test_size = 0.1 ,random_state=SEED)
    for train_ind,test_ind in strat.split(X,y):
        original_Xtrain, original_Xtest = X.iloc[train_ind], X.iloc[test_ind]
        original_ytrain, original_ytest = y.iloc[train_ind], y.iloc[test_ind]
    
    original_Xtest["class"] = original_ytest # create a test set
    
    return original_Xtrain, original_ytrain, original_Xtest.reset_index(drop=True)

def remove_features(df):
    """Function which removes unnecessary features. Used after pre-processing"""
    # Drop unnecessary and weakly classifying features
    df = df.drop(["eventId","accountNumber","mcc","merchantZip","merchantCountry","transactionAmount_binned","availableCash_binned"],axis=1)
    
    # Drop the year feature as it is not necessary
    df = df.drop(["year"],axis=1)
    
    # Drop time and date visualisation features
    df = df.drop(["day","hour","month"],axis=1)
    
    # Drop transaction time and posEntryMode as they are already encoded
    df = df.drop(["posEntryMode"],axis=1)

    return df.reset_index(drop=True)

def standardise_features(df):
    """Function which standardises 'transactionAmount', 'availableCash' and generates the 'transactionPercentage' features"""
    def standardise(x):
        mean = x.mean()
        std = x.std()
        
        return pd.Series((x-mean)/std)
    
    # Standardise transaction amount and available cash features
    df["transactionAmount"] = df["transactionAmount"].apply(standardise,by_row=False)
    df["availableCash"] = df["availableCash"].apply(standardise,by_row=False)
    df["transactionPercentage"] = df["transactionAmount"].abs()/df["availableCash"]
    return df

def merchantId_trust_score(df,test):
    """Function which incorporates historical data from the training set about merchant trustworthiness based on their previous history with frauds.
    Merchants who only appear once and are fraudulent can only be scored as 0.51 or 0.01 depending on whether their only interaction has been a fraud.
    More frequently occuring merchants are allowed to have a score anywhere from 0.01 to 1.00. The higher the number the worse.
    
    The test set is incorporated with knowledge from the train set - this means the test set classes are not touched. If a merchantId matches the trainset one its score will be taken.
    If the merchant exists only in the test set, it will be given a default score of 0.01.

    Args:
        df (pd.DataFrame): Training DataFrame which contains "merchantId"
        test (pd.DataFrame): Test DataFrame which contains "merchantId"

    Returns:
        df (pd.DataFrame): Scored training DataFrame with "merchantId" removed
        test (pd.DataFrame): Scored test DataFrame with "merchantId" removed
    """
    # Merchant Id aggregate class statistics
    # In this case mean will represent the proportion of class==1 as class can be only 1 or 0
    mercid_class_stat = df.groupby("merchantId")["class"].aggregate(["count","mean","min","max"])

    def _norm_score(x):
        # Normalize score into a different range with non-zero minimum
        if x["count"] == 1:
            return x["mean"] * 0.5 + 0.01 # base score of 0.01 is derived from class distribution in merchants which appear only once
        elif x["count"] >= 2 and x["count"] <= 20 :
            return x["mean"] * 0.99 + 0.01 
        else:
            return x["mean"] * 0.999 + 0.001
        
    def _map_scores(x, norm_scores):
        # Map each merchant to a score
        return norm_scores[x["merchantId"]]
    
    def _map_scores_test(x,norm_scores):
        # Map each merchant in the test set to score from the train set
        # Analysis shows 9703 (total of 4574 unique ids) of test set observations would have been affected by historical norm_score 
        try:
            return norm_scores[x["merchantId"]]
        except:
            return 0.5
            

    mercid_class_stat["norm_score"] = mercid_class_stat.apply(_norm_score,axis=1)
    df["norm_score"] = df.apply(_map_scores,args=(mercid_class_stat["norm_score"],),axis=1)
    test["norm_score"] = test.apply(_map_scores_test,args=(mercid_class_stat["norm_score"],),axis=1)
    return df.drop(["merchantId"],axis=1), test.drop(["merchantId"],axis=1)

# Sampling techniques
def random_undersampling(X_train,y_train):
    """Function which performs random undersampling"""
    # Undersample majority class to equalise classes
    rand_sampler = RandomUnderSampler(random_state=SEED)
    X_resample, y_resample = rand_sampler.fit_resample(X_train,y_train)
    X_resample, y_resample = X_resample.reset_index(drop=True), y_resample.reset_index(drop=True)
    return X_resample, y_resample

def near_miss_sampling(X_train,y_train):
    """Function which performs Near-miss sampling"""
    # Undersample majority class to equalise classes
    nearmiss_samp = NearMiss(n_jobs=-1,sampling_strategy="majority")
    X_resample, y_resample = nearmiss_samp.fit_resample(X_train,y_train)
    X_resample, y_resample = X_resample.reset_index(drop=True), y_resample.reset_index(drop=True)
    return X_resample, y_resample

def smote_sampling(X_train,y_train):
    """Function which performs SMOTE oversampling. The majority class data is purposefully undersampled first, after which the oversmapling is performed. This results in less synthetic cases and therefore
    less chances for non-representative samples"""
    
    # Create a smaller subset of the negative class to perform less oversampling of minority class (~ 2x)
    X_train_copy = X_train.copy()
    X_train_copy["class"] = y_train
    X_train_subset = X_train_copy.loc[X_train_copy["class"]==0].sample(2000,random_state=SEED)
    X_train_subset = pd.concat([X_train_subset,X_train_copy.loc[X_train_copy["class"]==1]],axis=0)
    
    y_train_subset = X_train_subset["class"]
    X_train_subset = X_train_subset.drop(["class"],axis=1)
    
    # Oversamples
    smote_sampler = SMOTE(sampling_strategy="minority",random_state=SEED,n_jobs=-1)
    X_resample, y_resample = smote_sampler.fit_resample(X_train_subset,y_train_subset)
    X_resample, y_resample = X_resample.reset_index(drop=True), y_resample.reset_index(drop=True)
    return X_resample, y_resample

# EDA including justifications and explanations
def analysis(df,print_analysis = PRINT_ANALYSIS):
    
    def missing_data(data):
        # MISSING DATA
        total = data.isnull().sum().sort_values(ascending=False)
        percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
        missing = pd.concat([total,percent],axis=1,keys=["Total","Percent"]).transpose()
        print("MISSING VALUES BY COUNT AND PERCENTAGE: ","\n",missing,"\n\n")
        # There is only one column with missing values and its merchantZip. Around 20% of the values are missing

        
    if print_analysis:
        #Check missing data
        missing_data(original_dataset)
        
        print("PROPORTION OF NON-FRAUD VS FRAUD TRANSACTIONS IN THE DATASET: ","\n",df["class"].value_counts(normalize=True),"\n\n")
        # Roughly 1 : 99 in terms of fraud: non-fraud ratio
        # The data will be undersampled to 50:50
        
        #ACCOUNT NUMBER
        # Account number is a feature which has the potential to contain a lot of useful information about
        # a transaction. Each account holder is different and as such, some might be uniquely susceptible to fraud
        # However, this means that in order to make use of it, a separate ML algorithm is needed
        # to categorise each account (using unsupervised methods) or to train fit a model specific to the account's tendiencies, 
        # In this case we will attempt to create an account-agnostic model which attempts to
        # predict fraud based on features which do not describe account-specific tendencies 
        
        #YEAR
        print("PROPORTION OF FRAUDS BY YEAR:",
            "\n",
            df.loc[df["class"]==1]["year"].value_counts(normalize=True),
            "\n\n")
        temp = df.loc[df["year"]==2017]
        print("PROPORTION OF NON-FRAUD VS FRAUD TRANSACTIONS IN THE DATASET (2017 ONLY)","\n",temp["class"].value_counts(normalize=True),"\n\n")
        # Removing 2018 from the dataset will remove less than 5% of the total fraud exampels.
        # The ratio of fraud to non-fraud remains close to 1:99 after removing 2018
        
        #MERCHANT
        print("NUMBER OF UNIQUE MERCHANT IDS: ", len(df["merchantId"].unique()))
        
        # Single instance
        val_c = df["merchantId"].value_counts()
        selected = val_c[val_c==1].index
        temp = df[df["merchantId"].isin(selected)]
        temp = temp.reset_index(drop=True).iloc[:,1:]
        print("CLASS DISTRIBUTION IN SINGLE INSTANCE MERCHANTS: ", "\n", temp["class"].value_counts())
        print("NUMBER OF DATA EXAMPLES WHERE MERCHANT IDs APPEARS ONLY ONCE: ",
            sum(val_c[val_c==1]),
            "\n\n")
        
        # 10 or more instances
        selected = val_c[val_c>=10].index
        temp = df[df["merchantId"].isin(selected)]
        temp = temp.reset_index(drop=True).iloc[:,1:]
        print("CLASS DISTRIBUTION IN >=10 INSTANCES MERCHANTS: ", "\n", temp["class"].value_counts())
        print("NUMBER OF DATA EXAMPLES WHERE MERCHANT IDs APPEAR 10 OR MORE TIMES: ",
            sum(val_c[val_c>=10]),          
            "\n\n")
        
        # 100 or more instances
        selected = val_c[val_c>=100].index
        temp = df[df["merchantId"].isin(selected)]
        temp = temp.reset_index(drop=True).iloc[:,1:]
        print("CLASS DISTRIBUTION IN >=100 INSTANCE MERCHANTS: ", "\n", temp["class"].value_counts())
        print("NUMBER OF DATA EXAMPLES WHERE MERCHANT IDs APPEAR 100 OR MORE TIMES: ",
            sum(val_c[val_c>=100]),
            "\n\n")
        # Nearly 61% of all the data has merchant IDs which appear only once. If any meaningful information is to be learned from this feature we will need ones which appear semi-frequently in the data. 
        # There is 0.2% (26092 data points) which have merchantIds which appear 100 or more times and ~5% (60764) which appear 10 or more times.
        # The class distributions for these merchantIDS dont show any reasonable classification power so the feature will be dropped
            
        print("NUMBER OF UNIQUE MERHCANT COUNTRIES: ", len(df["merchantCountry"].unique()))
        val_c_countries = df["merchantCountry"].value_counts(normalize=True)
        print("TOP 5 COUNTRIES' PROPORTIONS ADDED: ",val_c_countries.head(5).sum(),"\n\n") # Top 5 countries amount to 97.2% of the total data
        
        print("NUMBER OF UNIQUE MERHCANT CATEGORY CODES: ",len(df["mcc"].unique()))
        val_c = df["mcc"].value_counts(normalize=True)
        total = 0
        i = 1
        while total <= 0.95:
            total = sum(val_c.head(i))
            i += 1
        print(f"{i} MINIMUM CATEGORIES NECESSARY TO COVER AT LEAST {total*100:.2f}% OF THE DATA", "\n")
        # This is a categorical variable with 361 total unique values. As seen in the plots, the distribution of this feature is much more unifrom than merchant countries.
        # Consequently, the analysis shows to cover at least 95% of the data a minimum of 103 categories are required. This feature is further analysed in the plots section but ultimately dropped. 
        
        #POS
        print("UNIQUE VALUES IN POS ENTRY MODE: ", df["posEntryMode"].unique())
        print("NUMBER OF DATA EXAMPLES PER CATEGORY: ","\n", df["posEntryMode"].value_counts(normalize=True))
        print("UNKNOWN CATEGORY 79 CLASS DISTRIBUTION: ","\n" ,df.loc[df["posEntryMode"]==79]["class"].value_counts(normalize=True),"\n\n")
        # The POS feature contains an unknown category "79" which contains only non-fraud cases. Therefore, since "79" is not present in the data dictionary and is effectively unknown it can be
        # mapped to the "0" category
    else:
        pass

def plot(df,plot=PLOT):
    if plot:
        fig,axes = plt.subplots(1,3,figsize=(15,5))
        fig.suptitle("DISTRIBUTION OF HOURS, DAYS AND MONTHS PER CLASS")
        sns.kdeplot(data=df.loc[df["class"]==0],x="hour",color="blue",ax=axes[0])
        sns.kdeplot(data=df.loc[df["class"]==1],x="hour",color="red",ax=axes[0])
        axes[0].set_xlabel("Hour")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Density plot of transactions by hour and by class (Blue = Non-fraud | Red = Fraud)")
        # Plot shows that there are clearly different distributions for the fraudulent and non-fradulent
        # transactions with respect to the hour of the day. There is a sharp spike around 10 and less frauds than non-frauds later into the night and earlier into the day
        # this means the hour of the day has some information encoded into it that might be useful for classification
    
        sns.kdeplot(data=df.loc[df["class"]==0],x="day",color="blue",ax=axes[1])
        sns.kdeplot(data=df.loc[df["class"]==1],x="day",color="red",ax=axes[1])
        axes[1].set_xlabel("Day")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Density plot of transactions by day and by class (Blue = Non-fraud | Red = Fraud)")
        #Plot shows a near unifrom distribution for the non-fradulent transactions and a sharp spike towards the end of the month for fraduelent ones
        # This could perhaps be explained as increased fraduelent activity due to proximity to the average pay day
        # again, this means it might provide useful information to the model for classification
        
        sns.kdeplot(data=df.loc[(df["class"]==0) & (df["year"]==2017)],x="month",color="blue",ax=axes[2])
        sns.kdeplot(data=df.loc[(df["class"]==1) & (df["year"]==2017)],x="month",color="red",ax=axes[2])
        axes[2].set_xlabel("Month")
        axes[2].set_ylabel("Density")
        axes[2].set_title("Density plot of transactions by month and by class (Blue = Non-fraud | Red = Fraud)")
        # Plot shows a near unifrom distribution for non-fraduelent activity with oscilliations which are at exactly 1 per month.
        # The reason for these oscillations is due to there being only 12 values for the months and the KDE interpolating between each month
        # There is a large spike for non-fradulent activity during january but this is likely due to the timespan of the data which includes january twice.
        # This data from 2018 should be removed as to not influence the model due to a changed distribution. Otherwise, january transactions will be much more likely to be false negative.
        # The fraud distribution shows a distinct increase in June to October with a slight decrease in August. 
        # Meanwhile the non-fraud trans. seem to increase slightly in August.  
    
        fig,axes = plt.subplots(1,2,figsize=(15,5))
        fig.suptitle("BOXPLOTS OF TRANSACTION AMOUNTS PER CLASS")
        sns.boxplot(ax=axes[0],data=df,x="class",hue="class",y="transactionAmount")
        sns.boxplot(ax=axes[1],data=df,x="class",hue="class",y="transactionAmount",showfliers=False)
        axes[0].set_xlabel("CLASS")
        axes[1].set_xlabel("CLASS")
        print("Non Fraud transaction amount statistics")
        print(df[["class","transactionAmount"]].loc[df["class"]==0]["transactionAmount"].describe(),"\n\n")
        
        print("Fraud transaction amount statistics")
        print(df[["class","transactionAmount"]].loc[df["class"]==1]["transactionAmount"].describe(),"\n\n")
        # Fig.1 shows that non-fraud trans. have much higher extremal values. (max of non-fraud is >2x that of fraud)
        # Fig 2 shows that both fraud and non-fraud trans. have the same median but fraud has higher mean and std
        # The IQR of fraud trans is larger; slightly lower Q1 and significantly higher Q3.
        # The min. whiskers are near equal, whilst the upper max. whisker is significantly higher for fraud. 
        # There is discriminatory information present in the transactionalAmount feature
    
        fig,axes= plt.subplots(1,3,figsize=(15,5))
        fig.suptitle("STATISTICS OF THE TRANSACTION AMOUNT PERCENTAGE OF THE TOTAL AVAILABLE CASH")
        temp = df[["class","transactionAmount","availableCash"]].copy()
        temp["transactionPercentage"] = temp["transactionAmount"].abs()/temp["availableCash"]
        sns.kdeplot(ax=axes[0],data=temp.loc[temp["class"]==0],x="transactionPercentage",color="blue")
        sns.kdeplot(ax=axes[0],data=temp.loc[temp["class"]==1],x="transactionPercentage",color="red")
        sns.boxplot(ax=axes[1],data=temp,x="class",y="transactionPercentage",hue="class")
        sns.boxplot(ax=axes[2],data=temp,x="class",y="transactionPercentage",hue="class",showfliers=False)

        # The graph shows both fraud and non-fraud transactions tend to be centered at low percentages
        # However the fraud distribution has a slightly longer tail and lower extramal values as seen previously
        # The boxplots show that fraudulent transactions are significantly different from the non-fraudluent ones and in general, a higher proportion of the available cash
        # This means Transaction percetantage has some classification power and can be used
    
        fig,axes = plt.subplots(2,2,figsize=(15,5))
        fig.suptitle("TOTAL AMOUNT OF AVAILABLE CASH STATISTICS")
        sns.histplot(ax= axes[0,0],data=df.loc[df["class"]==0],x="availableCash",stat="percent",bins=10,color="blue")
        sns.histplot(ax=axes[0,0],data=df.loc[df["class"]==1],x="availableCash",stat="percent",bins=10,color="red")
        sns.kdeplot(ax=axes[0,1],data=df.loc[df["class"]==0],x="availableCash",color="blue")
        sns.kdeplot(ax=axes[0,1],data=df.loc[df["class"]==1],x="availableCash",color="red")
        sns.boxplot(ax=axes[1,0],data=df,x="class",hue="class",y="availableCash")
        sns.boxplot(ax=axes[1,1],data=df,x="class",hue="class",y="availableCash",showfliers=False)
        # The density plot of available cash per class shows that accounts with lower cash are at a higher risk of 
        # fraud. There is a clear discrepancy between the two classes - signals show that accounts with more than 5000 available cash are
        # much less likely to expereience fraud. There are a multitude of reasons for this such as lower income may mean less skilled work
        # and therefore less financially educated or less aware of dangerous activity.
        # This feature has good classification power
        
        fig,axes = plt.subplots(2,2,figsize=(15,5))
        fig.suptitle("COUNTRY BUCKETING AUGMENTATION STATISTICS")
        temp = df.copy()
        top5 = temp["merchantCountry"].value_counts().head(5).index
        key = {i:new_ind for new_ind,i in enumerate(top5)}
        # As seen previously in the analysis section, there are many unique countries in this dataset however the top 5 alone account for
        # >97% of the total data points. This means that the rest of the countries can be summarised into a single class which represents rare countries
        
        # Plots before augmentation
        sns.kdeplot(ax=axes[0,0],data=temp.loc[temp["class"]==0],x="merchantCountry",color="blue")
        sns.kdeplot(ax=axes[0,1],data=temp.loc[temp["class"]==1],x="merchantCountry",color="red")
        axes[0,0].set_title("BEFORE AUGMENTATION (NON-FRAUD)")
        axes[0,1].set_title("BEFORE AUGMENTATION (FRAUD)")
        
        # Augment the "merchantCountry" feature with the mapping function and plot the density again
        dataAugmented = temp.apply(keepOnlyTop5Countries,args=(top5,key,),axis=1)
        temp["merchantCountry"]= dataAugmented
        sns.kdeplot(ax=axes[1,0],data=temp.loc[temp["class"]==0],x="merchantCountry",color="blue")
        sns.kdeplot(ax=axes[1,1],data=temp.loc[temp["class"]==1],x="merchantCountry",color="red")
        axes[1,0].set_title("AFTER AUGMENTATION (NON-FRAUD)")
        axes[1,1].set_title("AFTER AUGMENTATION (FRAUD)")
        # The graph shows that after augmentation there are much clearer, sharper peaks and discrepancies between the two classes. 
        # One particular case is the rare-country class - 5. It shows a large difference between the fraud and non-fraud cases. It should add a substantial amount of classifying power to the model.
        
        fig,axes = plt.subplots(1,1,figsize=(15,5))
        fig.suptitle("MERCHANT CATEGORY STATISTICS")
        sns.kdeplot(ax = axes, data=df.loc[df["class"]==0],x="mcc",color="blue")
        sns.kdeplot(ax = axes, data=df.loc[df["class"]==1],x="mcc",color="red")
        # The plot shows that mcc contains some classification power particularly in the range from 4000-5500. In the range 4500-4800 there is a spike of fraudulent data points 
        # and at 5000-5300 there is a sharp spike of non-fradulent points. This could be used however the rest of the distribution is largely similar. This means that if we apply the same augmentation as 
        # merchant country the density of the new category will be very similar between the two classes and therefore very little classification power will be extracted. This feature will be dropped.
        
        fig,axes = plt.subplots(1,1,figsize=(15,5))
        fig.suptitle("POINT OF SALE ENTRY MODE STATISTICS")
        # temp = df.copy()
        # dataAugmented = temp.apply(pos_mapping,axis=1)
        # temp["posEntryMode"]= dataAugmented
        
        sns.kdeplot(ax=axes,data=temp.loc[temp["class"]==0],x="posEntryMode",color="blue")
        sns.kdeplot(ax=axes,data=temp.loc[temp["class"]==1],x="posEntryMode",color="red")
        # Category 3 ( 5 in the original dataset) shows a strong signal. Moreover, it is both the most predominant category present in the dataset and it is also pre-dominantly associated with non-fraud cases.
        # This means that this category will be a strong non-fraud classifier. The other two signals at 1 and 6 are shared with the fraud cases, however the fraud distribution's signals are broader. That means
        # there are more fraud cases in categories 1,2,5,7 and 8.
        plt.tight_layout()
        plt.show()
    else:
        pass

def plot_roc_pr_curve(y_test,y_hat,model):
    """Function which calculates FPR,TPR and the optimal threshold of a model based on Youden's J statistic."""
    
    # Calculate FPR and TPR at different thresholds
    fpr ,tpr, thresholds = roc_curve(y_test,y_hat)
    
    # Calculate J stat
    Y_J_statistic = tpr-fpr
    indx_J = np.argmax(Y_J_statistic)
    
    print(f"Best threshold J-statistic: {thresholds[indx_J]}")
    
    # Plot the ROC curve
    plt.figure()
    plt.plot([0,1],[0,1],linestyle="--",label="cut-off line")
    plt.plot(fpr,tpr,marker=".",label="ROC")
    plt.scatter(fpr[indx_J],tpr[indx_J],marker="o",color="black",label=f"Best threshold (J-statistic)\nFPR: {fpr[indx_J]} - TPR: {tpr[indx_J]}")

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    
    # Save the figure
    plt.savefig(f"results/roc_curve_{model}")
    df_results = {"thresholds":thresholds,"fpr":fpr,"tpr":tpr}
    
    # Return the optimal threshold and FPR-TPR table 
    return thresholds[indx_J], pd.DataFrame.from_dict(df_results)
    
# Training scripts
def train_logreg(X_train,y_train,X_test,y_test,sample_type="smote_oversample"):
    """Training function which performs a grid search on a logistic regression model and evaluates the model using a stratified k-fold cross validation.
        The performance plots are saved to the 'results' folder.
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.DataFrame): Training target labels
        X_test (pd.DataFrame): Test feature data
        y_test (pd.DataFrame): Test target labels
        sample_type (str, optional): The type of sampling technique to use. Can be either of ["smote_oversample","near_miss_undersample","random_undersample"]. Defaults to "smote_oversample".

    Returns:
        scores (dict): Dict of the training, validation and test cross validation results 
    """
    #Define performance metrics
    scores = {"LR_train_precision":[],"LR_train_f1":[],"LR_train_auc":[],"LR_train_recall":[],
              "LR_cv_precision":[],"LR_cv_f1":[],"LR_cv_auc":[],"LR_cv_recall":[],
              "LR_test_precision":[],"LR_test_f1":[],"LR_test_auc":[],"LR_test_recall":[],}
    
    sampler = {"random_undersample":random_undersampling,
               "near_miss_undersample":near_miss_sampling,
               "smote_oversample":smote_sampling}
    
    # Create a training and validation set and standarise the validation set
    X_train, y_train, valid = stratified_split(X_train,y_train)
    X_train = X_train.iloc[:,1:]
    valid = valid.iloc[:,1:]
    valid = standardise_features(valid)
    
    def _refit_strategy(cv_results):
        """Function which returns the index of the maximum of the geometric mean of the F1, AUC and precision cross validation scores during grid search. Used to find best candidate."""
        cv_results_ = pd.DataFrame(cv_results)
        mean_test_scores = cv_results_[
            ["mean_test_f1", 
             "mean_test_roc_auc",
             "mean_test_precision",]
        ]
        
        # Geometric mean of F1, AUC and Precision
        mean_test_scores = mean_test_scores.fillna(0)
        mean_test_scores["wavg"] = np.exp(np.log(mean_test_scores).mean(axis=1))
        return mean_test_scores["wavg"].idxmax()
    
    # Initialise a Logistic Regression model
    log_reg = LogisticRegression(random_state=SEED)
    
    # Sample to 50:50 class distribution
    X_standardised = standardise_features(X_train) 
    X_sampled,y_sampled = sampler[sample_type](X_standardised, y_train)
    
    X_sampled = X_sampled.values
    y_sampled = y_sampled.values
    
    # Define parameters to search through 
    log_reg_params = {"max_iter":[1e4,1e5,1e6],
                      "tol":[1e-4,1e-8],
                      "solver": ["liblinear","newton-cholesky"], 
                      'C': [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1, 10, 1e2, 1e3,1e4,1e5,1e6]}
    
    # Run Grid Search to find best hyperparams
    grid_log_reg = GridSearchCV(log_reg, log_reg_params,n_jobs=-1,verbose=1,cv = 5, scoring= ["f1","roc_auc","precision"],refit=_refit_strategy)
    grid_log_reg.fit(X_sampled, y_sampled)
    
    # Get the best logistic regression parameters
    best_params = grid_log_reg.best_params_
    print("Best parameters: ")
    pprint(best_params)
    log_reg = grid_log_reg.best_estimator_
    log_reg = log_reg.fit(X_sampled,y_sampled)
    
    # Confusion matrix for the train set
    preds = log_reg.predict(X_sampled)
    ConfusionMatrixDisplay.from_predictions(y_sampled, preds)
    plt.savefig("results/train_CM_LR.png")
    cm = metrics.confusion_matrix(y_sampled, preds)
    print("Train set @ 0.5","\n",cm)
    
    #Confusion matrix for the validation set
    X_valid = valid.drop(["class"],axis=1)
    y_valid = valid["class"]
    preds_proba = log_reg.predict_proba(X_valid)[:,1]
    best_threshold_J, res_df = plot_roc_pr_curve(y_valid,preds_proba,model="LR")
    print(res_df)
    
    preds = np.where(preds_proba>=best_threshold_J,1,0)
    ConfusionMatrixDisplay.from_predictions(y_valid, preds)
    plt.savefig("results/valid_CM_LR.png")
    cm = metrics.confusion_matrix(y_valid, preds)
    print(f"Validation set (J-Statistic @ {best_threshold_J})","\n",cm)
    
    # Confusion matrix for the test set
    preds_proba = log_reg.predict_proba(X_test)[:,1]
    preds = np.where(preds_proba>=best_threshold_J,1,0)
    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.savefig("results/test_CM_LR.png")
    cm = metrics.confusion_matrix(y_test, preds)
    print(f"Test set (J-Statistic @ {best_threshold_J})","\n",cm)
    
    print("Grid Search Finished ... Starting 5-fold cross validation with train-set sampling ")
    kfold = StratifiedKFold(n_splits=5,random_state=SEED,shuffle=True)
    # Sample during cross validation to avoid data leakeage
    for train_idx,valid_idx in tqdm(kfold.split(X_train,y_train)):
        
        X_train_split, X_valid_split = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_split, y_valid_split = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        # Standardise the data and remove unncessary features
        X_train_split_std = standardise_features(X_train_split) 
        X_valid_split_std = standardise_features(X_valid_split)

        # Sample the data
        X_train_sampled,y_train_sampled = sampler[sample_type](X_train_split_std, y_train_split)

        # Turn data into arrays
        X_train_sampled = X_train_sampled.values
        X_valid_split_std = X_valid_split_std.values
        y_train_sampled = y_train_sampled.values
        y_valid_split = y_valid_split.values
        
        # Define and train the Logistic regression model with the best parameters from GSCV
        log_reg = LogisticRegression(max_iter=best_params["max_iter"],tol=best_params["tol"],penalty="l2",solver=best_params["solver"],C=best_params["C"],random_state=SEED)
        model = log_reg.fit(X_train_sampled,y_train_sampled)
        
        # Training set - Sampled (50:50), Threshold .5
        preds = model.predict(X_train_sampled)
        scores["LR_train_precision"].append(precision_score(y_train_sampled,preds))
        scores["LR_train_recall"].append(recall_score(y_train_sampled,preds))
        scores["LR_train_f1"].append(f1_score(y_train_sampled,preds))
        scores["LR_train_auc"].append(roc_auc_score(y_train_sampled,preds))
        
        # Cross validation set - Unsampled (1:99 class dist), Best Threshold 
        preds_proba = model.predict_proba(X_valid_split_std)[:,1]
        preds = np.where(preds_proba>=best_threshold_J,1,0)
        scores["LR_cv_precision"].append(precision_score(y_valid_split,preds))
        scores["LR_cv_recall"].append(recall_score(y_valid_split,preds))
        scores["LR_cv_f1"].append(f1_score(y_valid_split,preds))
        scores["LR_cv_auc"].append(roc_auc_score(y_valid_split,preds))
        
        # Test set - Unsampled (1:99 class dist), Best Threshold
        preds_proba = model.predict_proba(X_test)[:,1]
        preds = np.where(preds_proba>=best_threshold_J,1,0)
        scores["LR_test_precision"].append(precision_score(y_test,preds))
        scores["LR_test_recall"].append(recall_score(y_test,preds))
        scores["LR_test_f1"].append(f1_score(y_test,preds))
        scores["LR_test_auc"].append(roc_auc_score(y_test,preds))
    return scores

def gradboost(X_train,y_train,X_test,y_test,sample_type="smote_oversample",rand_search=False):
    """Training function which performs a halved random search using a XGClassifier model and evaluates the model using a stratified k-fold cross validation.
        The performance plots are saved to the 'results' folder.
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.DataFrame): Training target labels
        X_test (pd.DataFrame): Test feature data
        y_test (pd.DataFrame): Test target labels
        sample_type (str, optional): The type of sampling technique to use. Can be either of ["smote_oversample","near_miss_undersample","random_undersample"]. Defaults to "smote_oversample".
        rand_search (bool,optinal): `True` to perform random search. Disable if you wish to use previously found parameters for the seed used. Defaults to False.

    Returns:
        scores (dict): Dict of the training, validation and test cross validation results 
    """
    # Define scoring metrics
    scores = {"XGB_train_precision":[],"XGB_train_f1":[],"XGB_train_auc":[],"XGB_train_recall":[],
              "XGB_cv_precision":[],"XGB_cv_f1":[],"XGB_cv_auc":[],"XGB_cv_recall":[],
              "XGB_test_precision":[],"XGB_test_f1":[],"XGB_test_auc":[],"XGB_test_recall":[]}
    
    sampler = {"random_undersample":random_undersampling,
               "near_miss_undersample":near_miss_sampling,
               "smote_oversample":smote_sampling}
    
    # Define the random search parameter spaces
    params = {
        "n_estimators":randint(500,10001),
        "learning_rate":  uniform(0,1),
        "max_delta_step":randint(1,10),
        "grow_policy":["depthwise","lossguide"],
        'gamma': uniform(0,1e2),
        'max_depth': randint(6,11),
        'min_child_weight': randint(1,101),
        'subsample': uniform(0.1,0.9),
        'colsample_bytree': uniform(0.1,0.9),
        "lambda": uniform(0,1e2),
        "alpha": uniform(0,1e2),
        "num_parallel_tree":randint(1,11)
        }
    
    # Create a training and validation set and standarise the validation set
    X_train, y_train, valid = stratified_split(X_train,y_train)
    X_train = X_train.iloc[:,1:]
    valid = valid.iloc[:,1:]
    valid = standardise_features(valid) 
    
    # Standardise and sample train search set
    X_search_std = standardise_features(X_train) 
    X_search_sampled,y_search_sampled = sampler[sample_type](X_search_std, y_train)

    #Create validation arrays
    X_valid = valid.drop(["class"],axis=1).values
    y_valid = valid["class"].values

    # Turn search set into arrays
    X_search_sampled = X_search_sampled.values
    print(f"Class distribution of train data after sampling with {sample_type}: ", "\n",y_search_sampled.value_counts())
    y_search_sampled = y_search_sampled.values
    
    # Random Search - performed only if argument is given, otherwise use a previously found best set of parameters to save time
    if rand_search:
        print("Running HalvingRandomSearch with 5-fold cross validation... Might take long")
        model =  xgb.XGBClassifier(objective="binary:logistic",random_state=SEED,verbosity=1,max_leaves=0) 
        search = HalvingRandomSearchCV(model,param_distributions=params,scoring="roc_auc",n_jobs=4,verbose=2,random_state=SEED)
        search.fit(X_search_sampled,y_search_sampled)
        best_params = search.best_params_
        print("Best parameters: ")
        pprint(best_params)
        model = search.best_estimator_.fit(X_search_sampled,y_search_sampled)
    else:
        print(f"Using previously found best parameter values for SEED=2041999")
        # Found during last search 
        best_params = { 
            'alpha': 3.556605507582389,
            'colsample_bytree': 0.4376901940771508,
            'gamma': 27.648075181070553,
            'grow_policy': 'lossguide',
            'lambda': 20.68844194394629,
            'learning_rate': 0.34478098903829424,
            'max_delta_step': 8,
            'max_depth': 10,
            'min_child_weight': 16,
            'n_estimators': 8614,
            'num_parallel_tree': 8,
            'subsample': 0.39115385632667665}
        print("Best parameters: ", best_params)
        
        model =  xgb.XGBClassifier(objective="binary:logistic",
                                   random_state=SEED,
                                   verbosity=1,
                                   max_leaves=0,
                                   reg_alpha=best_params["alpha"],
                                   colsample_bytree=best_params["colsample_bytree"],
                                   gamma = best_params["gamma"],
                                   grow_policy=best_params["grow_policy"],
                                   learning_rate=best_params["learning_rate"],
                                   max_delta_step=best_params["max_delta_step"],
                                   max_depth=best_params["max_depth"],
                                   min_child_weight=best_params["min_child_weight"],
                                   n_estimators=best_params["n_estimators"],
                                   num_parallel_tree=best_params["num_parallel_tree"],
                                   subsample=best_params["subsample"], 
                                   reg_lambda=best_params["lambda"])
        
        print("Fitting sampled data...")
        model.fit(X_search_sampled,y_search_sampled)
 
    # Confusion matrix for the train set
    preds = model.predict(X_search_sampled)
    ConfusionMatrixDisplay.from_predictions(y_search_sampled, preds)
    cm = metrics.confusion_matrix(y_search_sampled, preds)
    print("Train set @ 0.5")
    print(cm)
    plt.savefig("results/train_CM_XGB.png")
    
    
    # Confusion matrix for the validation set
    preds_proba = model.predict_proba(X_valid)[:,1]
    best_threshold_j,res_df = plot_roc_pr_curve(y_valid,preds_proba,model="XGB") 
    print(res_df)
    preds = np.where(preds_proba>best_threshold_j,1,0)
    ConfusionMatrixDisplay.from_predictions(y_valid, preds)
    plt.savefig("results/valid_CM_XGB.png")
    cm = metrics.confusion_matrix(y_valid, preds)
    print(f"Validation Set (J-Statistic @ {best_threshold_j})")
    print(cm)
    
    # Confusion matrix for the test set
    preds_proba = model.predict_proba(X_test)[:,1]
    preds = np.where(preds_proba>=best_threshold_j,1,0)
    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.savefig("results/test_CM_XGB.png")
    cm = metrics.confusion_matrix(y_test, preds)
    print(f"Test set (J-Statistic @ {best_threshold_j})","\n",cm)
    
    print("Grid Search Finished ... Starting 5-fold cross validation with train-set sampling ")
    kfold = StratifiedKFold(n_splits=5,random_state=SEED,shuffle=True)
    for train_idx,valid_idx in tqdm(kfold.split(X_train,y_train),"Cross validation with train-set oversampling. Can be slow (~40s/iter)"):
        X_train_split, X_valid_split = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_split, y_valid_split = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        # Standardise the data and remove unncessary features
        X_train_split_std = standardise_features(X_train_split) 
        X_valid_split_std = standardise_features(X_valid_split)
        
        # Sample the data
        X_train_sampled,y_train_sampled = sampler[sample_type](X_train_split_std, y_train_split)

        # Turn data into arrays
        X_train_sampled = X_train_sampled.values
        X_valid_split_std = X_valid_split_std.values
        y_train_sampled = y_train_sampled.values
        y_valid_split = y_valid_split.values

        # Fit the model on the sampled X train split
        model.fit(X_train_sampled,y_train_sampled)
        
        # Training set - Sampled (50:50), Threshold .5
        preds_train = model.predict(X_train_sampled)
        scores["XGB_train_precision"].append(precision_score(y_train_sampled,preds_train))
        scores["XGB_train_recall"].append(recall_score(y_train_sampled,preds_train))
        scores["XGB_train_f1"].append(f1_score(y_train_sampled,preds_train))
        scores["XGB_train_auc"].append(roc_auc_score(y_train_sampled,preds_train))
        
        # Validation set - Unsampled (1:99 class dist), Threshold @ Best found
        preds_proba = model.predict_proba(X_valid_split_std)[:,1]
        preds_cv = np.where(preds_proba>best_threshold_j,1,0)
        scores["XGB_cv_precision"].append(precision_score(y_valid_split,preds_cv))
        scores["XGB_cv_recall"].append(recall_score(y_valid_split,preds_cv))
        scores["XGB_cv_f1"].append(f1_score(y_valid_split,preds_cv))
        scores["XGB_cv_auc"].append(roc_auc_score(y_valid_split,preds_cv))
        
        # Validation set - Unsampled (1:99 class dist), Threshold @ Best found
        preds_proba = model.predict_proba(X_test)[:,1]
        preds_cv = np.where(preds_proba>best_threshold_j,1,0)
        scores["XGB_test_precision"].append(precision_score(y_test,preds_cv))
        scores["XGB_test_recall"].append(recall_score(y_test,preds_cv))
        scores["XGB_test_f1"].append(f1_score(y_test,preds_cv))
        scores["XGB_test_auc"].append(roc_auc_score(y_test,preds_cv))
        
    # Plot and save feature importance plot
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    xgb.plot_importance(model,height=0.8,title="Feature Imporatance" ,ax=ax, color="blue")
    plt.savefig("results/XGB_Feature_importance.png")
    return scores

# Evaluation scripts
def plot_evaluation_scores(LR_scores, XGB_scores):
    """Function which plots the final cross-validation and test results and saves them to 'results' folder""" 
    
    #Test and scores
    df_LR = pd.DataFrame.from_dict(LR_scores)
    df_XGB = pd.DataFrame.from_dict(XGB_scores)
    
    #Save scores to CSV
    df_LR.to_csv("results/LR_CV_TEST_results.csv")
    df_XGB.to_csv("results/XGB_CV_TEST_results.csv")
    
    # Separate LR and XGB into 2 dataframes
    df_cv_LR = df_LR.iloc[:,4:8]
    df_test_LR = df_LR.iloc[:,8:]
    df_cv_XGB = df_XGB.iloc[:,4:8]
    df_test_XGB = df_XGB.iloc[:,8:]
    
    # Replace column names to matching ones
    for i in [df_cv_LR,df_cv_XGB]:
        i.columns = ["cv_precision","cv_f1","cv_auc","cv_recall"]
    for i in [df_test_LR,df_test_XGB]:
        i.columns = ["test_precision","test_f1","test_auc","test_recall"]
        
    # Plot CV results and save figure
    _,axes = plt.subplots(1,1,figsize=(15,5))
    sns.barplot(data=df_cv_LR.mean(),ax=axes,color="red",alpha=0.5)
    sns.barplot(data=df_cv_XGB.mean(),ax=axes,color="blue",alpha=0.5)
    plt.savefig("results/CV_performance.png")
    
    # Plot Test results and save figure
    _,axes = plt.subplots(1,1,figsize=(15,5))
    sns.barplot(data=df_test_LR.mean(),ax=axes,color="red",alpha=0.5)
    sns.barplot(data=df_test_XGB.mean(),ax=axes,color="blue",alpha=0.5)
    plt.savefig("results/TEST_performance.png")
    return 0

# Check if results folder exists
if not os.path.exists("./results"):
    os.mkdir("./results")
    
# Load data
original_dataset = load_dataset(regen=args.regen)
all_transactions = original_dataset.copy()

# General pre-processing
all_transactions = one_hot_encode_features(all_transactions)

if PLOT or PRINT_ANALYSIS:
    # Generate EDA plots and print analysis if PLOT and ANALYSIS are set to TRUE
    _X = all_transactions.drop(["class"],axis=1)
    _y = all_transactions["class"]
    _X_train, _y_train, _ = stratified_split(_X,_y)
    _X_train["class"] = _y_train
    plot(_X_train)
    analysis(_X_train)

all_transactions = remove_features(all_transactions)

# Separate a train-test set and standardise separately to avoid leakeage
X = all_transactions.drop(["class"],axis=1)
y = all_transactions["class"]
X_train, y_train, test = stratified_split(X,y) # test is roughly equal to monthly average transactions 9000 ish
X_train["class"] = y_train

# Process test data
test = standardise_features(test)
X_test = test.drop(["class"],axis=1).iloc[:,1:]
y_test = test["class"]

# Enable historical to add historical merchant "trust" score to the datasets
if HISTORICAL:
    X_train,X_test = merchantId_trust_score(X_train,X_test)
else:
    X_train = X_train.drop(["merchantId"],axis=1)
    X_test = X_test.drop(["merchantId"],axis=1)
    
# Remove outliers
X_train,_ = remove_outliers(X_train)
y_train = X_train["class"]
X_train = X_train.drop(["class"],axis=1)

# Train models
scores_LR_CV= train_logreg(X_train,y_train,X_test,y_test,sample_type=args.sampling)
pprint(scores_LR_CV)
scores_XGB_CV = gradboost(X_train,y_train,X_test,y_test,sample_type= args.sampling,rand_search=args.rand_search)
pprint(scores_XGB_CV)

# Create and save evaluation plots
plot_evaluation_scores(scores_LR_CV,scores_XGB_CV)
plt.show()
