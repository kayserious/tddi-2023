import modeler
import constants
import pandas as pd
import argparse, sys


parser = argparse.ArgumentParser()

parser.add_argument('--input_data', '-i', type= str)
parser.add_argument('--output_data', '-o', type= str, default= 'predictions.csv')

args = parser.parse_args()

df = pd.read_csv(args.input_data,sep = constants.SEPARATOR)

md = modeler.KAYSERIOUSModel(live=True)

md.raise_from_binary(binary_path = constants.SAVE_DEPLOYED_TO,text_column = constants.TEXT_NAME, target_column = constants.TARGET_NAME)

predictions = md.predict(df)

predictions.to_csv(args.output_data,sep = constants.SEPARATOR,index = False)