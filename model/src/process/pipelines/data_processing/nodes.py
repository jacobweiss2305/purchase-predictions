import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing


def add_revenue(clickstream: pd.DataFrame, purchase: pd.DataFrame) -> pd.DataFrame:
    """Add total revenue and binary revenue as response variables

    Args:
        clickstream (pd.DataFrame): Clickstream data
        purchase (pd.DataFrame): Purchase data

    Returns:
        pd.DataFrame: Feature table with response variables
    """
    purchase['revenue'] = purchase['price'] * purchase['quantity'] 
    total_revenue = purchase.groupby(level=0).revenue.sum()
    feature_table = clickstream.join(total_revenue).drop_duplicates()
    feature_table['binary_revenue'] = feature_table.revenue.notna().astype(int)
    feature_table['revenue'] = feature_table['revenue'].fillna(0)
    return feature_table

def total_number_purchases(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Total number of purchases by session ID

    Args:
        feature_table (pd.DataFrame): Features

    Returns:
        pd.DataFrame: Features with total number of purchases
    """    
    agg = feature_table.groupby(level=0).binary_revenue.sum()
    agg = agg.rename('total_purchases')
    feature_table = feature_table.join(agg)
    return feature_table

def category_map(clickstream: pd.DataFrame) -> pd.DataFrame:
    """Category mapping based on the following logic:
        - If an item has been clicked in the context of a promotion or special offer, then the value will be "S"
        - If the item has been clicked under a regular category (i.e., sport), the value will be a number between 1 to 12.
        - If the context was a brand (i.e., Nike, Adidas, ...), then the value will be an 8-10 digits number.
        - "0" indicates a missing value

    Args:
        clickstream (pd.DataFrame): Clickstream data

    Returns:
        pd.DataFrame: Mapping of category
    """    
    conditions = [(clickstream.category == 'S'),
                  (clickstream.category.isin([str(i) for i in np.arange(1,13,1)])),
                  (clickstream.category.str.match('\(|\)|\d{7}')),
                  ((clickstream.category == '0') | (clickstream.category == 'nan'))]

    choices = ['special offer','regular','brand', 'missing']
    clickstream['category_map'] = np.select(conditions, choices, default=np.nan)
    return clickstream['category_map']

def total_clicks(clickstream: pd.DataFrame) -> pd.DataFrame:
    """Count the total clicks per session

    Args:
        clickstream (pd.DataFrame): Clickstream data

    Returns:
        pd.DataFrame: Count of clicks per session
    """
    total_clicks = clickstream.groupby(level=0).timestamp.count()
    total_clicks = total_clicks.rename("total_clicks")
    return total_clicks


def total_seconds(clickstream: pd.DataFrame) -> pd.DataFrame:
    """Measure the total seconds per session

    Args:
        clickstream (pd.DataFrame): Clickstream data

    Returns:
        pd.DataFrame: seconds per session
    """
    first_occurance = clickstream.groupby(level=0).timestamp.first()
    last_occurance = clickstream.groupby(level=0).timestamp.last()
    seconds_per_session = pd.merge(first_occurance, last_occurance, left_index=True, right_index=True)
    last_time = pd.to_datetime(seconds_per_session.iloc[:, 1])
    first_time = pd.to_datetime(seconds_per_session.iloc[:, 0])
    seconds_per_session['seconds_per_session'] = (last_time - first_time).dt.total_seconds()
    return seconds_per_session


def count_unique_items(clickstream: pd.DataFrame) -> pd.DataFrame:
    """Count total unique items per session

    Args:
        clickstream (pd.DataFrame): Clickstream data

    Returns:
        pd.DataFrame: Count total unique items per session
    """
    unique_items = clickstream.groupby(level=0).item_id.nunique()
    unique_items = unique_items.rename("count_unique_items")
    return unique_items


def stat_plots(df: pd.DataFrame, column_name: str):
    print(df[column_name].describe())
    plt.subplot(1, 2, 1)
    ax = df[column_name].hist()
    ax.set_title(f"Distribution of {column_name}")
    ax.set_xlabel(f"{column_name}")
    ax.set_ylabel("Count")

    plt.subplot(1, 2, 2)
    ax = df.boxplot(column=column_name)
    ax.set_title(f"Distribution of {column_name}")
    ax.set_ylabel("Count")

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=1.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.show()


def stacked_bar_chart(df: pd.DataFrame, legend: str):
    x = df.columns
    y2 = df.iloc[1]
    plt.bar(x, y2, color='b')
    plt.title(legend)
    plt.show()


def standardize(df):
    values = ['total_clicks', 'seconds_per_session', 'count_unique_items']
    x = df[values]
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    scaled = pd.DataFrame(scaler.transform(
        x), columns=values).fillna(0).set_index(df.index)
    return df.drop(values, axis=1).join(scaled)
