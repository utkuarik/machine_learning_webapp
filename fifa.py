import streamlit as st
import pandas as  pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sys
from pandas.errors import ParserError
import time
import altair as alt
import matplotlib.cm as cm


st.title('Streamlit Demo')

class Predictor:

    def prepare_data(self, split_data):
        data = self.data[['overall', 'height_cm', 'weight_kg','age', 'potential']]
        data = data.sample(frac = round(split_data/100,2))
        target_options = data.columns
        self.chosen_target = st.sidebar.selectbox("Please choose target column", (target_options))
        X = data.loc[:, data.columns != self.chosen_target]
        y = data[self.chosen_target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        
    def set_classifier_properties(self):
        self.chosen_classifier = st.sidebar.selectbox("Please choose a classifier", ('Random Forest', 'Linear Regression')) 
        if self.chosen_classifier == 'Random Forest': 
            self.n_trees = st.sidebar.slider('number of trees', 1, 1000, 1)
     
            
    def predict(self, predict_btn):        
        if self.chosen_classifier == 'Random Forest':
            self.regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=self.n_trees)
            self.model = self.regr.fit(self.X_train, self.y_train)
            predictions = self.regr.predict(self.X_test)
            self.predictions = predictions
        
        elif self.chosen_classifier=='Linear Regression':
            self.regr = LinearRegression()
            self.model = self.regr.fit(self.X_train, self.y_train)
            predictions = self.regr.predict(self.X_test)
            self.predictions = predictions

        result = pd.DataFrame(columns=['Actual', 'Prediction'])
        result['Actual'] = self.y_test
        result['Prediction'] = predictions
        result.sort_index()
        self.result = result

        return self.predictions, self.result

    def get_metrics(self):
        self.error_metrics = {}
        self.error_metrics['MSE'] = mean_squared_error(self.y_test, self.predictions)

        return st.markdown('### MSE: ' + str(round(self.error_metrics['MSE'], 3)))


    def plot_result(self):
        fig, axes = plt.subplots(figsize=(12, 8))   
        axes.scatter(self.result.index, self.result.Actual, color ='#2300A8', label = 'Actual ' + str(self.chosen_target),
        alpha=0.70, cmap=cm.brg)
        axes.scatter(self.result.index, self.result.Prediction, color = '#00A658', label = 'Predicted ' + str(self.chosen_target),
        alpha=0.70, cmap=cm.brg)
        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(15) 
        for tick in axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(15) 
        axes.set_xlabel('Index', fontsize=20)
        axes.set_ylabel('Value',fontsize=20)
        axes.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        axes.legend(bbox_to_anchor=(1, 1), loc='2', borderaxespad=0. , prop={'size':20})
        st.pyplot()

        ## ALTAIR PLOT ##
        # self.result['Index'] = self.result.index
        # result = self.result.melt('Index')
        # c = alt.Chart(result).mark_circle(size=60).encode(
        # x='Index',
        # y='value',
        # color = 'variable'
    
        
        # ).properties(
        #     width= 650,
        #     height=450 )

        # st.altair_chart(c + c)

    def file_selector(self, folder_path='.'):
        filenames = os.listdir(folder_path)
        selected_filename = st.sidebar.selectbox('Select a file', filenames)
        return folder_path, selected_filename
    
    @st.cache()
    def read_file(self,  selected_filename, folder_path='.'):
        return pd.read_csv(os.path.join(folder_path, selected_filename))

    def print_table(self):
        if len(self.result) > 0:
            # print_checkbox = st.sidebar.checkbox('Show results as a table')
            # if print_checkbox:
            st.dataframe(self.result.style.highlight_max(axis=0))
            
    
if __name__ == '__main__':

    controller = Predictor()
    try:
        selected_filename, folder_path = controller.file_selector()
        controller.data = controller.read_file(folder_path, selected_filename)
        split_data = st.sidebar.slider('Randomly divide data %', 1, 100, 10 )
        controller.prepare_data(split_data)
        controller.set_classifier_properties()
    except (AttributeError, ParserError, KeyError) as e:
        st.markdown('<span style="color:blue">WRONG FILE TYPE</span>', unsafe_allow_html=True)  

    
    
    predict_btn = st.sidebar.button('Predict')

    predictions, result = controller.predict(predict_btn)
    if predict_btn:
        controller.get_metrics()
        controller.plot_result()
  


    controller.print_table()

    if st.sidebar.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    if st.sidebar.checkbox('Show histogram'):
        chosen_column = st.selectbox("Please choose a columns", ('Value', 'Overall', 'Potential'))   
        st.subheader('Histogram')
        plt.hist(player_list[chosen_column], bins=20, edgecolor='black')
        st.pyplot()






