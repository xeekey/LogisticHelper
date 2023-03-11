import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense


# load and preprocess the data
def load_and_preprocess_data(file_path, cat_cols, num_cols):
    # check if preprocessed data and encoders exist in file
    preprocessed_data_file = 'preprocessed_data.joblib'
    if os.path.exists(preprocessed_data_file):
        X, y, cat_encoder, num_scaler = joblib.load(preprocessed_data_file)
    else:
        # load the data from the CSV file
        data = pd.read_csv(file_path)

        # remove any rows that contain NaN values
        data = data.dropna()

        # remove any rows where the VAEGT value contains ".00" and convert to integer
        data['Weight'] = data['Weight'].apply(lambda x: int(x.split('.')[0]))

        # remove leading/trailing spaces from column names
        data.columns = data.columns.str.strip()

        # one-hot encode categorical variables
        cat_encoder = OneHotEncoder()
        X_cat = cat_encoder.fit_transform(data[cat_cols])

        # scale numerical variables
        num_scaler = StandardScaler()
        X_num = num_scaler.fit_transform(data[num_cols])

        # combine categorical and numerical variables
        X = np.hstack([X_cat.toarray(), X_num])

        # extract the Weight column as the target variable
        y = data['Weight']

        # save preprocessed data and encoders to file
        joblib.dump((X, y, cat_encoder, num_scaler), preprocessed_data_file)

    return X, y, cat_encoder, num_scaler

# train and evaluate the model
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_path, cat_encoder, num_scaler):
    # define the neural network model
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='linear'))

    # compile the model with mean squared error loss and Adam optimizer
    model.compile(loss='mse', optimizer='adam')

    # train the model on the training data
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    # evaluate the model on the testing data
    score = model.evaluate(X_test, y_test, verbose=0)

    # save the trained model
    save_model(model, model_path)

    return model, score

# load a trained model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# preprocess input data and make predictions
def preprocess_prediction_data(pred_data, cat_encoder, num_scaler, model):
    # preprocess the new data
    cat_cols = ['ProductName', 'FORMAT']
    num_cols = ['Amount']
    X_pred_cat = cat_encoder.transform(pred_data[cat_cols])
    X_pred_num = num_scaler.transform(pred_data[num_cols])
    X_pred_processed = np.hstack([X_pred_cat.toarray(), X_pred_num])

    # use the model to make predictions
    y_pred = model.predict(X_pred_processed)
    predicted_weights_transformed = y_pred.flatten()
    predicted_weights = num_scaler.inverse_transform(predicted_weights_transformed.reshape(-1, 1)).flatten()

    return predicted_weights



# example usage
if __name__ == '__main__':
    cat_cols = ['ProductName', 'FORMAT']
    num_cols = ['Amount']

    model_path = 'my_model.h5'
    
    X, y, cat_encoder, num_scaler = load_and_preprocess_data('shipment_data.csv', cat_cols, num_cols)
    if os.path.exists(model_path):
        loaded_model = load_trained_model(model_path)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        model, score = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_path, cat_encoder, num_scaler)
        loaded_model = model

    
    X_pred = pd.DataFrame({'FORMAT': ['150 cm - Kæmpe (150 x 160-220 cm)', '200 cm - Extreme (200 x 170-300 cm m. teleskopstæn', '148 x 222 mm'],
                           'ProductName': ['Roll-up', 'Roll-up', 'Idem sæt'],
                           'Amount': [55551, 15555, 5000000]})
    print(preprocess_prediction_data(X_pred, cat_encoder, num_scaler, loaded_model) / 10000)




# from imp import load_module, load_source
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential, save_model, load_model
# from tensorflow.keras.layers import Dense


# def load_and_preprocess_data(file_path, cat_cols, num_cols):
#     # load the data from the CSV file
#     data = pd.read_csv(file_path) 


#     data = pd.read_csv(file_path)

#     # remove any rows that contain NaN values
#     data = data.dropna()

#     # remove any rows where the VAEGT value contains ".00" and convert to integer
#     data['Weight'] = data['Weight'].apply(lambda x: int(x.split('.')[0]))


#     # remove leading/trailing spaces from column names
#     data.columns = data.columns.str.strip()

#     # one-hot encode categorical variables
#     cat_encoder = OneHotEncoder()
#     X_cat = cat_encoder.fit_transform(data[cat_cols])

#     # scale numerical variables
#     num_scaler = StandardScaler()
#     X_num = num_scaler.fit_transform(data[num_cols])

#     # combine categorical and numerical variables
#     X = np.hstack([X_cat.toarray(), X_num])

#     # extract the Weight column as the target variable
#     y = data['Weight']

#     return X, y, cat_encoder, num_scaler


# def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_path):
#     # define the neural network model
#     model = Sequential()
#     model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dense(1, activation='linear'))

#     # compile the model with mean squared error loss and Adam optimizer
#     model.compile(loss='mse', optimizer='adam')

#     # train the model on the training data
#     model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
#    # evaluate the model on the testing data
#     score = model.evaluate(X_test, y_test, verbose=0)

#     # save the trained model
#     save_model(model, model_path)

#     return model, score

# def load_trained_model(model_path):
#     model = load_model(model_path)
#     return model



# def preprocess_prediction_data(pred_data, cat_encoder, num_scaler, model):
#     # preprocess the new data
#     cat_cols = ['ProductName', 'FORMAT']
#     num_cols = ['Amount']
#     X_pred_cat = cat_encoder.transform(pred_data[cat_cols])
#     X_pred_num = num_scaler.transform(pred_data[num_cols])
#     X_pred_processed = np.hstack([X_pred_cat.toarray(), X_pred_num])

#     # use the model to make predictions
#     y_pred = model.predict(X_pred_processed)
#     predicted_weight_transformed = y_pred[0][0]
#     predicted_weight = num_scaler.inverse_transform([[predicted_weight_transformed]])[0][0]

#     return predicted_weight

# # load and preprocess the data
# cat_cols = ['ProductName', 'FORMAT']
# num_cols = ['Amount']
# #X, y, cat_encoder, num_scaler = load_and_preprocess_data('shipment_data.csv', cat_cols, num_cols)

# #plit the data into training and testing sets
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# #train and evaluate the model
# model_path = 'my_model.h5'
# cat_encoder = OneHotEncoder()
# num_scaler = StandardScaler()
# #model, score = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_path)

# loaded_model = load_trained_model(model_path)

# #make predictions on new data
# X_pred = pd.DataFrame({'FORMAT': ['150 cm - Kæmpe (150 x 160-220 cm)'],
#                        'ProductName': ['Roll-up'],
#                        'Amount': [1]})

# print(preprocess_prediction_data(X_pred, cat_encoder, num_scaler, loaded_model))