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



st.title('Streamlit Demo')

class Predictor:

    def prepare_data(self):
        #data = data[['short_name', 'value_eur','overall','potential','nationality']]
        # data['Value'] = data['Value'].apply(lambda x: x.replace("€", ""))  
        # data['Value'] = data['Value'].apply(lambda x: x.replace("M", "00000") if "." in x else x.replace("M", "000000"))
        # data['Value'] = data['Value'].apply(lambda x: x.replace("K", "000"))  
        # data['Value'] = data['Value'].apply(lambda x: x.replace(".", ""))  
        # data['Value'] = data['Value'].astype("double") 
        data = self.data[['overall', 'height_cm', 'weight_kg','age', 'potential']]
        target_options = data.columns
        self.chosen_target = st.sidebar.selectbox("Please choose target column", (target_options))
        X = data.loc[:, data.columns != self.chosen_target]
        y = data[self.chosen_target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        


    def predict(self, predict_btn):


        if predict_btn:

            if chosen_classifier == 'Random Forest':
                regr = RandomForestRegressor(max_depth=2, random_state=0)
                model = regr.fit(self.X_train, self.y_train)
                predictions = regr.predict(self.X_test)
                self.predictions = predictions
                st.write('Model MSE is: ' + str(self.get_metrics() ))
            elif chosen_classifier=='Linear Regression':
                regr = LinearRegression()
                model = regr.fit(self.X_train, self.y_train)
                predictions = regr.predict(self.X_test)
                self.predictions = predictions

                st.write('Model MSE is: ' + str(self.get_metrics() ))
            
            result = pd.DataFrame(columns=['Actual', 'Prediction'])
            result['Actual'] = self.y_test
            result['Prediction'] = predictions
            result.sort_index()

            self.result = result
            return self.predictions, self.result

    def get_metrics(self):
        self.error_metrics = {}
        self.error_metrics['MSE'] = mean_squared_error(self.y_test, self.predictions)

        return self.error_metrics


    def plot_result(self):
        figure = plt.figure()
        axes = figure.add_axes([0, 0, 0.7, 0.7])
        axes.scatter(self.result.index, self.result.Actual, color ='red', label = 'Actual ' + str(self.chosen_target))
        axes.scatter(self.result.index, self.result.Prediction, color = 'blue', label = 'Predicted ' + str(self.chosen_target))
        axes.set_xlabel('Index')
        axes.set_ylabel('Value')
        axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        st.pyplot()

    def file_selector(self, folder_path='.'):
        filenames = os.listdir(folder_path)
        selected_filename = st.sidebar.selectbox('Select a file', filenames)
        self.data = pd.read_csv(os.path.join(folder_path, selected_filename))
        return self.data


        


if __name__ == '__main__':
    
    controller = Predictor()
    controller.file_selector()
    controller.prepare_data()
    chosen_classifier = st.sidebar.selectbox("Please choose a classifier", ('Random Forest', 'Linear Regression'))  
    predict_btn = st.sidebar.button('Predict')
    if predict_btn:
        
        predictions, result = controller.predict(predict_btn)
        controller.get_metrics()
        controller.plot_result()




    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    if st.checkbox('Show histogram'):
        chosen_column = st.selectbox("Please choose a columns", ('Value', 'Overall', 'Potential'))   
        st.subheader('Histogram')
        plt.hist(player_list[chosen_column], bins=20, edgecolor='black')
        st.pyplot()






