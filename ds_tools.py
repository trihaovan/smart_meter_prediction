import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def basic_eda(df):
    """
    Carries out a basic EDA on the input DataFrame and prints a summary.

    Args:
        df (DataFrame)
    
    """
    rows = df.shape[0]
    cols = df.shape[1]
    dups = df.duplicated().sum()
    nulls = df.isna().sum().sum()
    null_cols = [i for i in df.isna().sum()[df.isna().sum()!=0].index]
    dups_pct = round(100*dups/rows, 2)

    print(
        f"====================== BASIC EDA ======================\
        \nRows: {rows}\
        \nColumns: {cols}\
        \nDuplicates: {dups} rows, {dups_pct}% of the DataFrame\
        \nNulls: {nulls} null values {'in ' if nulls != 0 else ''}{null_cols}"
    )

######################################################################################################################### 

def vect_and_plot (df_in, vect = 'CV', tokens = None, mindf = 50, maxdf = 1.0, ngrams = (1,1)):
    """
    Returns fitted bag of words, transformed input, and token counts
    Plots distribution of token occurrences, and top and bottom 20 tokens

    Inputs:
        df_in: DataFrame to be vectorised
        vect: String, 'CV' for CountVectorizer or 'Tfid' for TfidVectorizer
        tokens: Tokeniser, default = None
        min_df: int, Minimum number of token occurrences
        max_df: int, Maximum number of token occurrences
        ngrams: Range of possible ngrams

    Returns:
        BoW_fitted: Fitted bagofwords
        df_transformed: Transformed input DF
        count_df: Token counts
    """

    from sklearn.feature_extraction.text import CountVectorizer

    # Instantiate the CountVectorizer if 'CV' input
    if vect == 'CV':
        from sklearn.feature_extraction.text import CountVectorizer
        BoW = CountVectorizer(tokenizer = tokens, min_df = mindf, max_df = maxdf, ngram_range = ngrams)

    # Instantiate TfidfVectorizer if 'Tfidf' input
    elif vect == 'Tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer                               
        BoW = TfidfVectorizer(tokenizer = tokens, min_df = mindf, max_df = maxdf, ngram_range = ngrams)
    
    else:
        return print('Vectoriser must be CountVectorizer (CV) or TfidfVectorizer (Tfidf)')
        
    # Fit and transform positive and negative reviews
    BoW_fitted = BoW.fit(df_in)
    df_transformed = BoW_fitted.transform(df_in)
    idx = BoW_fitted.get_feature_names()

    count_df = pd.DataFrame(
    {'counts':df_transformed.toarray().sum(axis = 0)}, 
    index = idx
    ).sort_values('counts', ascending = False)

    # Plot distribution of counts
    plt.figure(figsize = (15,7))
    sns.histplot(count_df, bins = 100, log_scale=True, legend = False, color = 'cornflowerblue')
    sns.despine()
    plt.xlabel('Token Occurrences')
    plt.title('Distribution of Token Occurrences')
    plt.show()

    # Plot top 20
    plt.figure(figsize = (15,7))
    sns.barplot(data = count_df.head(20), x = count_df.head(20).index, y = 'counts', color = 'cornflowerblue')
    sns.despine()
    plt.title("Top 20 Tokens")
    plt.ylabel("Occurrences")
    plt.xlabel('Token')
    plt.xticks(rotation=45)
    plt.show()

    # Plot bottom 20
    plt.figure(figsize = (15,7))
    sns.barplot(data = count_df.tail(20), x = count_df.tail(20).index, y = 'counts', color = 'cornflowerblue')
    sns.despine()
    plt.title("Bottom 20 Tokens")
    plt.ylabel("Occurrences")
    plt.xlabel('Token')
    plt.xticks(rotation=45)
    plt.show()

    print(f"=========================================================================\
          \n============= Vectorisation has produced {df_transformed.shape[1]} features =============\
          \n=========================================================================")
    # return fitted BoW, transformed input, and dataframe of counts
    return BoW_fitted, df_transformed, count_df

#########################################################################################################################
def PlotBoundaries(model, X, Y, figsize=(8, 6)):
    '''
    Helper function that plots the decision boundaries of a model and data (X,Y)
    code modified from: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
    '''

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.4)

    # Plot
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
    plt.show()

########################################################################################################

def decompose_DFs (df, mult_or_add, p):
    '''
    Function to decompose time series data and create an output dataframe of decomposed data

    ---INPUTS---
    df: Time series DataFrame
    mult_or_add: String, is TS data additive or multiplicative?
    p: Integer, seasonal period

    ---OUTPUT---
    output: DataFrame of decomposed time series

    '''
    from statsmodels.api import tsa

    output = pd.DataFrame()
    for col in df.columns:
        name = col + '_Decomp'
        globals()[name] = tsa.seasonal_decompose(df[[col]], model = mult_or_add, period = p, extrapolate_trend=True) # extrapolate_trend = 'freq' to fill in nulls
        output[col + '_Trend'] = globals()[name].trend
        output[col + '_Seasonal'] = globals()[name].seasonal
        output[col + '_Residual'] = globals()[name].resid
    
    return output