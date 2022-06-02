import json
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
locations = None
col_names = None
our_model = None


def get_predicted_price(location, sqft, bedrooms, bath):
    try:
        loc_index = col_names.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(col_names))
    x[0] = bath
    x[1] = sqft
    x[2] = bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
    return our_model.predict([x])[0]/100000


def Save_data_loaded():
    print("Loading saved artifacts....start")
    global col_names
    global locations
    global our_model

    with open('./artif/cols.json', 'r') as f:
        col_names = json.load(f)['col_name']
        locations = col_names[3:]
    global our_model
    if our_model is None:
        with open('./artif/Karachi_model.pickle', 'rb') as f:
            our_model = pickle.load(f)
    print("loading saved artifacts....done")


def get_location_names():
    return locations


def get_data_columns():
    return col_names


if __name__ == '__main__':
    Save_data_loaded()
    print(get_location_names())
    print(get_predicted_price('Nazimabad', 1800, 4, 3))
    print(get_predicted_price('Abid Town', 1388.48, 3, 3))
