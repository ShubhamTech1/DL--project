# ANN FOR REGRESSION:-

    
import pandas as pd 
df = pd.read_csv(r"D:\360DigiTMG\DATA SCIENTIST learning\DATA SCIENCE\ASSIGNMENTS\SUPERVISED LEARNING\REGRESSION\2 Multiple linear regression\50_Startups.csv")

'''
# MySQL Database connection
# Creating engine which connect to MySQL
user = 'user1' # user name
pw = 'user1' # password
db = 'startup_db' # database

from sqlalchemy import create_engine
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database 
data.to_sql('startup_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from startup_tbl'
df = pd.read_sql_query(sql, con = engine) 
'''

df.shape
df.dtypes
df.info()     # not any null values are present in our dataset.
df.describe() 
 
df.duplicated().sum()  
# not any duplicated rows are present here.

# now check any outliers are present or not :
import seaborn as sns
sns.boxplot(df)
# another method
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
# here we see not any outliers are present here.


# Seperating input and output variables 
X = pd.DataFrame(df.iloc[:, 0:4])  
y = pd.DataFrame(df.iloc[:, -1]) 


# Checking for unique values
X["State"].unique()     
X["State"].value_counts() 



# Segregating Non-Numeric features
categorical = X.select_dtypes(include = ['object']).columns 
print(categorical)
# Segregating Numeric features
numerical = X.select_dtypes(exclude = ['object']).columns
print(numerical)


## Missing values Analysis
# Define pipeline for missing data if any
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))]) 
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numerical)]) 

# Fit the imputation pipeline to input features
imputation = preprocessor.fit(X)

import joblib
# Save the pipeline
joblib.dump(imputation, 'Meanimpute')  ## Missing values

# Transformed data
clean = pd.DataFrame(imputation.transform(X), columns = numerical)
clean



## Outlier Analysis :
# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

X.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 


# Increase spacing between subplots
import matplotlib.pyplot as plt 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

from feature_engine.outliers import Winsorizer
# Winsorization for outlier treatment
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(clean.columns))

winsor
clean1 = winsor.fit(clean)





# Save winsorizer model
joblib.dump(clean1, 'Winsor')
cleandata = pd.DataFrame(clean1.transform(clean), columns = numerical)

# Boxplot
cleandata.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Scaling
## Scaling with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scale_pipeline = Pipeline([('scale', MinMaxScaler())])
columntransfer = ColumnTransformer([('scale', scale_pipeline, numerical)]) 
scale = columntransfer.fit(cleandata)


# Save Minmax scaler pipeline model
joblib.dump(scale, 'Minmax')
scaled_data = pd.DataFrame(scale.transform(cleandata), columns = numerical)
scaled_data.describe()



## Encoding
# Categorical features
from sklearn.preprocessing import OneHotEncoder 
encoding_pipeline = Pipeline([('onehot', OneHotEncoder())])
preprocess_pipeline = ColumnTransformer([('categorical', encoding_pipeline, categorical)])
clean2 =  preprocess_pipeline.fit(X)   # Works with categorical features only

# Save the encoding model
joblib.dump(clean2, 'Encoding')
encode_data = pd.DataFrame(clean2.transform(X))


# To get feature names for Categorical columns after Onehotencoding 
encode_data.columns = clean2.get_feature_names_out(input_features = X.columns)
encode_data.info()

clean_data = pd.concat([scaled_data, encode_data], axis = 1)  # concatenated data will have new sequential index
clean_data.info()






# NOW BUILD ANN MODEL
import tensorflow as tf
from tensorflow import keras

# Define the model
model = keras.Sequential([
  keras.layers.Dense(16, activation="relu", input_shape=(clean_data.shape[1],)),  # Hidden layer with 16 neurons and ReLU activation
  keras.layers.Dense(1, activation="linear")  # Output layer with 1 neuron and linear activation for regression
])


# Compile the model
model.compile(loss="mse", optimizer="adam", metrics=["mse"])

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Splitting data into training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(clean_data, y, test_size = 0.2, random_state = 0) 

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32) 

# Predict on the training data
Y_train_pred = model.predict(X_train) 

# Calculate the Mean Squared Error
mse = mean_squared_error(Y_train, Y_train_pred)
print("Training MSE:", mse)


# Make predictions on new data
predictions = model.predict(X_test)

# Calculate the Mean Squared Error
mse = r2_score(Y_test, predictions)  
print("Training MSE:", mse)































