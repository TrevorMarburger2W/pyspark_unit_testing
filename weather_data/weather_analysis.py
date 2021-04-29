from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StringType,
    BooleanType,
    DateType,
    IntegerType,
    DoubleType
)
import argparse


def cast_num_columns(data, cast_dict):
    '''Casts specified columns to Numeric types, returns transformed dataframe copy'''

    df = data

    if 'DoubleType' in cast_dict:
        for clm in cast_dict['DoubleType']:
            df = df \
                .withColumn(clm,col(clm).cast(DoubleType()))

    if 'IntegerType' in cast_dict:
        for clm in cast_dict['IntegerType']:
            df = df \
                .withColumn(clm,col(clm).cast(IntegerType()))

    return df



def core_logic(ctx, infile):
    '''Reads in data file provided in cli arg & performs data transforms'''

    df = ctx \
        .read \
        .format('csv') \
        .option("header", True) \
        .option("delimiter", ",") \
        .load(infile)

    df.printSchema()

    df = cast_num_columns(
        df,
        {
            'IntegerType': [
                'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Cloud9am'
                ],
            'DoubleType': [
                'MinTemp', 'MaxTemp', 'Rainfall', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Cloud9am', 'Cloud3pm'
                ]
        }
    )

    df.printSchema()



def app_main(infile):
    '''Creates SparkSession & Fires Core app logic'''

    ctx = SparkSession \
        .builder \
        .appName("SimpleApp") \
        .getOrCreate()

    core_logic(ctx, infile)
    

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        help="Fully qualified path of input CSV file."
    )
    args = parser.parse_args()
    if args.infile:
        INFILE = args.infile
    app_main(infile=INFILE)