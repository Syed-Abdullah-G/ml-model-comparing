import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout = "centered"
)

st.title("Compare Models Online")

#column configuration for graphs



# fetching the dataframe
got_column = "false"
train = st.file_uploader("Upload Training Data", type = ["csv"])
if train is not None:
    df = pd.read_csv(train)

    if got_column == "false":
        max_column = st.number_input("Maximum No of Unique Values to be allowed for a column",value = 15)
    else:
        pass


    # label encoding bool columns
    df_bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for i in df_bool_cols:
        df[i] = df[i].astype(int)

    # cat columns
    df_cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # dropping the Name like columns
    removed = []
    for i in df_cat_cols:
        unique_values_count = df[i].value_counts().count()
        if (unique_values_count) >max_column:
            df = df.drop(i, axis = 1)
            removed.append(i)


    # function to remove list from a list

    def remove_elements(original_list, elements_to_remove):
        return list(filter(lambda x: x not in elements_to_remove, original_list))
    # removing columns from df_cat_cols also (previously done in df)
    df_cat_cols = remove_elements(df_cat_cols, removed)
    
    # copying cat column names(without the initial ones and one hot encoded ones)
# one hot encoding

    for i in df_cat_cols:
        df_encoded = pd.get_dummies(df[i], prefix = i)
        df_encoded = df_encoded.astype(int)
        df = pd.concat([df, df_encoded], axis = 1)
        df = df.drop(i, axis=1)


    df.to_csv("final2.csv")


    def drop_nan(dataframe):
        if dataframe.isnull().values.any():
            dataframe = dataframe.dropna()
        return dataframe
    
    def fill_mean(dataframe):
        if dataframe.isnull().values.any():
            dataframe = dataframe.fillna(dataframe.mean())
        return dataframe
    


    
    df_drop = drop_nan(df)
    df_fill = fill_mean(df)


    # code block for droping null values
    df = df_drop
    
    y = st.radio(
                "Choose Target Variable",
                df.columns, index = None, key = "radio_1"
            )
    if st.button("Select the column", key = "button_1"):
        aux_y = df.copy()
        df.drop(y, axis =1)
        X_train, X_test, y_train, y_test = train_test_split(df, aux_y[y], test_size=0.2, random_state=42)
        # linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        lr_mse = mean_squared_error(y_test, y_pred)
        regr = RandomForestRegressor(random_state=0)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        rgr_mse = mean_squared_error(y_test, y_pred)
        labels = ["LR", "RFR"]
        result_for_drop = [lr_mse, rgr_mse]
        
        # code block for filling null values with mean
        df = df_fill
    
        y = y
        aux_y = df.copy()
        df.drop(y, axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(df, aux_y[y], test_size=0.2, random_state=42)
        # linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        lr_mse = mean_squared_error(y_test, y_pred)
        regr = RandomForestRegressor(random_state=0)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        rgr_mse = mean_squared_error(y_test, y_pred)
        labels = ["LR", "RFR"]
        result_for_fill = [lr_mse, rgr_mse]
        fig, ax  = plt.subplots(1)
        width = 0.35
        bar_positions_drop = np.arange(len(labels))
        bar_positions_fill = bar_positions_drop + width
        ax = plt.bar(bar_positions_drop,result_for_drop, width, label = "Drop")
        ax = plt.bar(bar_positions_fill,result_for_fill,width, label = "Fill Mean")
    

        plt.xlabel("ML Method")
        plt.ylabel("Mean Score Error")
        plt.legend(["Drop", "Fill Mean"])
        plt.title("Mean Score Error Comparison")
        plt.style.use("ggplot")
        plt.show()
        st.pyplot(fig, use_container_width = True)
        df_results = pd.DataFrame([[str(result_for_drop[0])+"  Drop", str(result_for_drop[1])+"  Drop"],
                                   [str(result_for_fill[0]) + "  Fill Mean", str(result_for_fill[1])+"  Fill Mean"]], columns=["Linear Regression", "Ran.Forest Regressor"])

        st.table(df_results)
        
        score = {
            "Linear Regression(Drop)": result_for_drop[0],
            "Random Forest Regressor(Drop)": result_for_drop[1],
            "Linear Regression(Fill Mean)": result_for_fill[0],
            "Random Forest Regressor(Fill Mean)": result_for_fill[1]
        }

        #defining a function to compute the minimum in a dictionary
        def find_min_value_and_key(dictionary):
            if not dictionary:
                return None, None  # Return None if the dictionary is empty

            # Use min function with a lambda function as the key to find the minimum value
            min_key, min_value = min(dictionary.items(), key=lambda x: x[1])

            return min_key, min_value

        min_key, min_value = find_min_value_and_key(score)
        
        st.header("Best Model:")
        st.write(min_key)
        st.success(min_value, icon="✅")
        
