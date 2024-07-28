"""
Tasks for Course: DLMDSPWP01 - Programming with Python (from IU International University of Applied Sciences)

Assignment Task:
- Task is to write a Python-program that uses training data to choose the four ideal functions which are the best fit out of the fifty provided.
- The criterion for choosing the ideal functions for the training function is how they minimize the sum of all y deviations  squared.
-  The criterion for mapping the individual test case to the four ideal functions is that the existing maximum deviation of the calculated regression does not exceed the largest deviation between training dataset and the ideal function chosen for it by more than  factor  sqrt(2).
- the program must use the test data provided to  determine for each and every x-y pair of values whether or not they can be assigned to the four chosen ideal functions; if so, the program also needs to execute the mapping and save it together with the deviation at hand.
- All data must be visualized logically

"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import column_property
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt


def main():
# Main function to run the data analysis framework.


    # Reads training, ideal, and test data from CSV files.
    trainCSV = './Datasets/train.csv'
    idealCSV = './Datasets/ideal.csv'
    testCSV = './Datasets/test.csv'

    # Identifies the ideal function for each training column.
    analysis = DataAnalysisOps(trainCSV, idealCSV, testCSV)
    analysis.read_data_from_csv()
    ideal_functions = analysis.identify_ideal_fn()

    # Calculates the maximum deviation for each ideal function.
    max_deviations = analysis.calculate_maximum_deviation()

    # Maps the test data to the ideal functions.
    mapped_data, merged_data = analysis.mapping_with_test_data(ideal_functions, max_deviations)

    # Creates SQLAlchemy tables for the training, ideal, and mapped test data.
    analysis.create_nd_store_tables(mapped_data)
    
    # Plots the training data, raw test data, and mapped test data.
    analysis.plotting(merged_data)
    print(mapped_data)


Base = declarative_base()

class TrainTable(Base):
# Defining struture of the train data table to be stored in SQLAlchemy

    __tablename__ = 'Traning Data'

    x = Column(Float, primary_key=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i in range(1, 5):
            setattr(self, f'y{i}', Column(Float))

    @classmethod
    def __declare_last__(cls):
        for i in range(1, 5):
            setattr(cls, f'y{i}', column_property(getattr(cls, f'y{i}')))

class IdealFnTable(Base):
# Defining struture of the Ideal functions table to be stored in SQLAlchemy

    __tablename__ = 'Ideal Functions'
    
    x = Column(Float, primary_key=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i in range(1, 51):
            setattr(self, f'y{i}', Column(Float))

    @classmethod
    def __declare_last__(cls):
        for i in range(1, 51):
            setattr(cls, f'y{i}', column_property(getattr(cls, f'y{i}')))

class TestTable(Base):
# Defining struture of the resultant mapped table to be stored in SQLAlchemy

    __tablename__ = 'Mapped Test Data'
    x = Column(Float, primary_key=True)
    y = Column(Float)
    delta_y = Column(Float)
    mapped_ideal_fn = Column(String)

class DataAnalysisFramework:
# Initializes the variables with the data to be processed. Loading data into the analytics system

    def __init__(self, trainCSV, idealCSV, testCSV) -> None:

        self.trainCSV = trainCSV
        self.idealCSV = idealCSV
        self.testCSV = testCSV
        self.trainingData = None
        self.idealData = None
        self.testData = None
    
    def read_data_from_csv(self) -> None:

        self.trainingData = pd.read_csv(self.trainCSV)
        self.idealData = pd.read_csv(self.idealCSV)
        self.testData = pd.read_csv(self.testCSV)
    
class DataAnalysisOps(DataAnalysisFramework):
    """
    Data analysis operations class.

    Inherits from the DataAnalysisFramework class.
    Provides additional methods for identifying ideal functions, calculating maximum deviations, mapping test data to ideal functions, creating SQLAlchemy tables, and plotting data.
    """
    
    def __init__(self, trainCSV, idealCSV, testCSV) -> None:

        super().__init__(trainCSV, idealCSV, testCSV)
    
    def identify_ideal_fn(self) -> dict:
        # Identifies the ideal function for each training column.
        
        if self.trainingData is None:
            raise Exception("Trainig data not loaded. Please load data as a prerequisite to running this operation")
        
        if self.idealData is None:
            raise Exception("Ideal data not loaded. Please load data as a prerequisite to running this operation")

        if self.testData is None:
            raise Exception("Test data not loaded. Please load data as a prerequisite to running this operation")
        
        ideal_fn = {}
        for train_col in self.trainingData.columns[1:]:
            temp = {}
            for ideal_col in self.idealData.columns[1:]:
                mse_values = mean_squared_error(self.trainingData[train_col], self.idealData[ideal_col])
                temp[ideal_col] = mse_values
            min_mse_fn = min(temp, key=temp.get)
            ideal_fn[train_col] = min_mse_fn
        return ideal_fn
    
    def calculate_maximum_deviation(self) -> dict:
        # Calculates the maximum deviation for each ideal function.

        max_deviations = {}
        for train_col, ideal_col in self.identify_ideal_fn().items():
            max_deviation = np.max(np.abs(self.trainingData[train_col] - self.idealData[ideal_col]))
            max_deviations[ideal_col] = max_deviation
        return max_deviations
    
    def mapping_with_test_data(self, ideal_fn:dict, max_deviations:dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Maps the test data to the ideal functions.

        ideal_fn_cols = list(ideal_fn.values())
        ideal_fn_cols.append('x')
        ideal_fn_rows = self.idealData[ideal_fn_cols]
        mergedData = self.testData.merge(ideal_fn_rows, on='x', how='left')

        mul_factor = math.sqrt(2)
        comparing_factor = {keys: values * mul_factor for keys, values in max_deviations.items()}

        result = {}
        for _, row in mergedData.iterrows():
            count = 0
            for col in mergedData.columns[2:]:
                deviation = abs(row['y'] - row[col])
                if deviation <= comparing_factor[col]:
                    result.setdefault('X', []).append(row['x'])
                    result.setdefault('Y', []).append(row['y'])
                    result.setdefault('Delta Y', []).append(deviation)
                    result.setdefault('Ideal Fn', []).append(col)
                    break
                elif count == 3:
                    result.setdefault('X', []).append(row['x'])
                    result.setdefault('Y', []).append(row['y'])
                    result.setdefault('Delta Y', []).append(deviation)
                    result.setdefault('Ideal Fn', []).append(None)
                count+=1

        mappedData = pd.DataFrame(result)
        return mappedData, mergedData
    
    def create_nd_store_tables(self, mappedData:pd.DataFrame) -> None:
        # Creates SQLAlchemy tables for the training, ideal, and mapped test data

        try:
            engine = create_engine('sqlite:///ListOfTables.db')
            self.trainingData.to_sql('Traning Data', con=engine, index=False, if_exists='replace')
            self.idealData.to_sql('Ideal Functions', con=engine, index=False, if_exists='replace')
            mappedData.to_sql('Mapped Test Data', con=engine, index=False, if_exists='replace')

        except:
            raise Exception("Error with DB operation.\nPlease check the following:\n1. Training and Ideal data is loaded\n2. Best ideal function mapping with Test data is complete")

    def plotting(self, mergedData:pd.DataFrame) -> None:
        # Plots the training data, raw test data, and mapped test data.
        
        for cols in self.trainingData.columns[1:]:
            plt.plot(self.trainingData['x'], self.trainingData[cols], label=f'Training Function {cols}', linewidth=1, linestyle=':')

        # Add labels and title
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.title('Traning Data Plot')
        # Add Legend
        plt.legend()
        # Show the plot
        plt.show()

        plt.plot(self.testData['x'], self.testData['y'], label='Test Data', linewidth=1, linestyle='--')

        # Add labels and title
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.title('Raw Test Data Plot')
        # Add Legend
        plt.legend()
        # Show the plot
        plt.show()

        for cols in mergedData.columns[1:]:
            plt.plot(mergedData['x'], mergedData[cols], label=f'{cols}', linewidth=1, linestyle='-.')

        # Add labels and title
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.title('Test Data merged with Ideal Functions Plotted')
        # Add Legend
        plt.legend()
        # Show the plot
        plt.show()

if __name__ == '__main__':
    main()