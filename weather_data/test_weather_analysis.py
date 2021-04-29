import logging
import unittest
from pyspark.sql import SparkSession
from weather_analysis import (
    cast_num_columns
)


class PySparkTest(unittest.TestCase):

    @classmethod
    def suppress_py4j_logging(cls):
        logger = logging.getLogger('py4j')
        logger.setLevel(logging.WARN)
    

    @classmethod
    def create_testing_pyspark_session(cls):
        return (
            SparkSession \
                .builder \
                .appName("local_test_ctx") \
                .getOrCreate()
        )
        

    @classmethod
    def setUp(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.create_testing_pyspark_session()


    @classmethod
    def tearDown(cls):
        cls.spark.stop()

    
class TestCasting(PySparkTest):

    def dummy_data_frame(self):
        '''Creates dummy weather DataFrame'''

        import pandas as pd
        import json
        
        # Create Test Data
        pandas_data = pd.DataFrame(
            {
                "Date":{"0":"2008-12-01","1":"2008-12-02","2":"2008-12-03","3":"2008-12-04","4":"2008-12-05"},
                "Location":{"0":"Albury","1":"Albury","2":"Albury","3":"Albury","4":"Albury"},
                "MinTemp":{"0":"13.4","1":"7.4","2":"12.9","3":"9.2","4":"17.5"},
                "MaxTemp":{"0":"22.9","1":"25.1","2":"25.7","3":"28","4":"32.3"},
                "Rainfall":{"0":"0.6","1":"0","2":"0","3":"0","4":"1"},
                "Evaporation":{"0":"0","1":"0","2":"0","3":"0","4":"0"},
                "Sunshine":{"0":"0","1":"0","2":"0","3":"0","4":"0"},
                "WindGustDir":{"0":"W","1":"WNW","2":"WSW","3":"NE","4":"W"},
                "WindGustSpeed":{"0":"44","1":"44","2":"46","3":"24","4":"41"},
                "WindDir9am":{"0":"W","1":"NNW","2":"W","3":"SE","4":"ENE"},
                "WindDir3pm":{"0":"WNW","1":"WSW","2":"WSW","3":"E","4":"NW"},
                "WindSpeed9am":{"0":"20","1":"4","2":"19","3":"11","4":"7"},
                "WindSpeed3pm":{"0":"24","1":"22","2":"26","3":"9","4":"20"},
                "Humidity9am":{"0":"71","1":"44","2":"38","3":"45","4":"82"},
                "Humidity3pm":{"0":"22","1":"25","2":"30","3":"16","4":"33"},
                "Pressure9am":{"0":"1007.7","1":"1010.6","2":"1007.6","3":"1017.6","4":"1010.8"},
                "Pressure3pm":{"0":"1007.1","1":"1007.8","2":"1008.7","3":"1012.8","4":"1006"},
                "Cloud9am":{"0":"8","1":"0","2":"0","3":"0","4":"7"},
                "Cloud3pm":{"0":"0","1":"0","2":"2","3":"0","4":"8"},
                "Temp9am":{"0":"16.9","1":"17.2","2":"21","3":"18.1","4":"17.8"},
                "Temp3pm":{"0":"21.8","1":"24.3","2":"23.2","3":"26.5","4":"29.7"},
                "RainToday":{"0":"No","1":"No","2":"No","3":"No","4":"No"},
                "RainTomorrow":{"0":"No","1":"No","2":"No","3":"No","4":"No"}
            }
        )

        return pandas_data

    def assert_frame_equal_with_sort(self, results, expected, keycolumns):
            '''Sorts two Pandas DataFrames to be compared & sorts on shared key - runs assert method'''

            from pandas.testing import assert_frame_equal

            results_sorted = results.sort_values(by=keycolumns).reset_index(drop=True)
            expected_sorted = expected.sort_values(by=keycolumns).reset_index(drop=True)

            assert_frame_equal(results_sorted, expected_sorted)

    def test_cast_num_columns(self):
        '''Tests cast_num_columns'''

        import pandas as pd
        from pandas.testing import assert_frame_equal
        import numpy as np

        spark_data = self.spark \
            .createDataFrame(self.dummy_data_frame())

        results_spark = cast_num_columns(
            spark_data,
            {
                'IntegerType': [
                    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Cloud9am'
                    ],
                'DoubleType': [
                    'MinTemp', 'MaxTemp', 'Rainfall', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Cloud9am', 'Cloud3pm'
                    ]
            }
        )
        results_pandas = results_spark.toPandas()
        expected_result = pd.DataFrame(
            {
                "Date":{"0":"2008-12-01","1":"2008-12-02","2":"2008-12-03","3":"2008-12-04","4":"2008-12-05"},
                "Location":{"0":"Albury","1":"Albury","2":"Albury","3":"Albury","4":"Albury"},
                "MinTemp":{"0":13.4,"1":7.4,"2":12.9,"3":9.2,"4":17.5},
                "MaxTemp":{"0":22.9,"1":25.1,"2":25.7,"3":28.0,"4":32.3},
                "Rainfall":{"0":0.6,"1":0.0,"2":0.0,"3":0.0,"4":1.0},
                "Evaporation":{"0":"0","1":"0","2":"0","3":"0","4":"0"},
                "Sunshine":{"0":"0","1":"0","2":"0","3":"0","4":"0"},
                "WindGustDir":{"0":"W","1":"WNW","2":"WSW","3":"NE","4":"W"},
                "WindGustSpeed":{"0":"44","1":"44","2":"46","3":"24","4":"41"},
                "WindDir9am":{"0":"W","1":"NNW","2":"W","3":"SE","4":"ENE"},
                "WindDir3pm":{"0":"WNW","1":"WSW","2":"WSW","3":"E","4":"NW"},
                "WindSpeed9am":{"0":20,"1":4,"2":19,"3":11,"4":7},
                "WindSpeed3pm":{"0":24,"1":22,"2":26,"3":9,"4":20},
                "Humidity9am":{"0":71,"1":44,"2":38,"3":45,"4":82},
                "Humidity3pm":{"0":22,"1":25,"2":30,"3":16,"4":33},
                "Pressure9am":{"0":1007.7,"1":1010.6,"2":1007.6,"3":1017.6,"4":1010.8},
                "Pressure3pm":{"0":1007.1,"1":1007.8,"2":1008.7,"3":1012.8,"4":1006.0},
                "Cloud9am":{"0":8,"1":0,"2":0,"3":0,"4":7},
                "Cloud3pm":{"0":0,"1":0,"2":2.0,"3":0,"4":8.0},
                "Temp9am":{"0":16.9,"1":17.2,"2":21.0,"3":18.1,"4":17.8},
                "Temp3pm":{"0":21.8,"1":24.3,"2":23.2,"3":26.5,"4":29.7},
                "RainToday":{"0":"No","1":"No","2":"No","3":"No","4":"No"},
                "RainTomorrow":{"0":"No","1":"No","2":"No","3":"No","4":"No"}
            }
        )
        int_cols = ['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Cloud9am']
        for int_col in int_cols:
            expected_result[int_col] = expected_result[int_col].astype('int32')
        

        
        self.assert_frame_equal_with_sort(
            results_pandas,
            expected_result,
            ['Date']
        )
        


if __name__ == '__main__':
    unittest.main()
