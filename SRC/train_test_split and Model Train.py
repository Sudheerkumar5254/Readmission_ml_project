# after consideration of 40 features out of 166 features. split the data into x_train,x_test,y_train,y_test.
#Based on historic information, admission info i am choosing these features for train the model.

# Patient_readmission=df1.copy()
#variables=['ageCat', 'gender', 'type.of.heart.failure', 'myocardial.infarction', 'congestive.heart.failure', 
#              'cerebrovascular.disease', 'diabetes', 'moderate.to.severe.chronic.kidney.disease', 'liver.disease', 
 #             'CCI.score', 'admission.ward', 'DestinationDischarge', 'discharge.department', 'visit.times', 
  #            'dischargeDay', 'acute.renal.failure', 'respiration', 'oxygen.inhalation', 
   #           'NYHA.cardiac.function.classification', 'Killip.grade', 'systolic.blood.pressure', 
    #          'diastolic.blood.pressure', 'body.temperature', 'BMI', 'LVEF', 'weight', 'height', 
     #         'creatinine.enzymatic.method', 'glomerular.filtration.rate', 'hemoglobin', 'platelet', 'D.dimer', 
      #        'high.sensitivity.troponin', 'brain.natriuretic.peptide', 'sodium', 'potassium', 'total.protein', 
       #       'cholesterol', 'triglyceride']
#target='re.admission.within.6.months'

x=Patient_readmission[variables]
y=Patient_readmission[target]

# check the target feature is balance or imbalnced by using value counts.
y.value_counts()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Handle missing values for categorical and numerical fetures in Train data and apply to test data.
# first u need to segregate the categorical and numerical fetures.

cate_cols = x_train.select_dtypes(include=['object']).columns
num_cols = x_train.select_dtypes(include=['number']).columns


# Handle missing values in categorical columns fill with mode
for col in cate_cols:
    mode_value = x_train[col].mode()[0] 
    x_train[col].fillna(mode_value, inplace=True)
    x_test[col].fillna(mode_value, inplace=True)

 # Handle missing values in numerical columns fill with median
for col in num_cols:
    median_value = x_train[col].median()   
    x_train[col].fillna(median_value, inplace=True)
    x_test[col].fillna(median_value, inplace=True)


# check and replace the outliers by IQR method.
# Train data after apply to test data
plt.figure(figsize=(12, 8))
plt.boxplot(x_train[num_cols])
plt.show() 

plt.figure(figsize=(12, 8))
plt.boxplot(x_test[num_cols])
plt.show()

# Apply IQR-based capping for numerical features
for col in num_cols:
    Q1 = x_train[col].quantile(0.25)
    Q3 = x_train[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Apply capping in train
    x_train[col] = np.where(x_train[col] < lower_bound, lower_bound, x_train[col])
    x_train[col] = np.where(x_train[col] > upper_bound, upper_bound, x_train[col])

    # Apply same capping in test
    x_test[col] = np.where(x_test[col] < lower_bound, lower_bound, x_test[col])
    x_test[col] = np.where(x_test[col] > upper_bound, upper_bound, x_test[col])

    # After apply the IQR method again we need to check outliers are there or not.

    plt.figure(figsize=(12, 8))
    plt.boxplot(x_train[num_cols])
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.boxplot(x_test[num_cols])
    plt.show()


# one hot encoding and Target encoding for categorical features.

# If there are 6 unqiue values of categorical feature use one hot encoding method.
from sklearn.preprocessing import OneHotEncoder

# One-Hot Encoding for remaining categorical variables
one_hot_encoding_vars = ['gender', 'type.of.heart.failure', 'admission.ward','DestinationDischarge', 'discharge.department',
                         'oxygen.inhalation','NYHA.cardiac.function.classification', 'Killip.grade']

ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

# Fit encoder on training categorical variables
x_train_ohe = ohe.fit_transform(x_train[one_hot_encoding_vars])
x_test_ohe = ohe.transform(x_test[one_hot_encoding_vars])

# Convert to DataFrame
x_train_ohe = pd.DataFrame(x_train_ohe, columns=ohe.get_feature_names_out(one_hot_encoding_vars))
x_test_ohe = pd.DataFrame(x_test_ohe, columns=ohe.get_feature_names_out(one_hot_encoding_vars))

# Reset index to match original dataset
x_train_ohe.index = x_train.index
x_test_ohe.index = x_test.index

# Drop original categorical columns and merge one-hot encoded columns
x_train = x_train.drop(columns=one_hot_encoding_vars).join(x_train_ohe)
x_test = x_test.drop(columns=one_hot_encoding_vars).join(x_test_ohe)

print("One-hot encoding applied successfully!")


# if more than 6 unique values are available in categorical feature by use Target encoding.
# Function to apply target encoding (mean of target per category)
def target_encode(train_col, target_col):
    target_mapping = train_col.groupby(train_col).apply(lambda x: target_col[x.index].mean())  # Compute mean target per category
    return train_col.map(target_mapping), target_mapping

# Apply target encoding to 'ageCat' for the training set
x_train['ageCat'], target_mapping = target_encode(x_train['ageCat'], y_train)

# Apply the same mapping on the test set (avoid data leakage)
x_test['ageCat'] = x_test['ageCat'].map(target_mapping)

# Handle unseen categories in the test set (fill NaN with the mean of target variable in training set)
x_test['ageCat'].fillna(x_train['ageCat'].mean(), inplace=True)

print(f" Target encoding applied on 'ageCat'!")


# Scaling the numerical features for all features equal contribute to the model.
num_cols = ['myocardial.infarction', 'congestive.heart.failure',
       'cerebrovascular.disease', 'diabetes',
       'moderate.to.severe.chronic.kidney.disease', 'liver.disease',
       'CCI.score', 'visit.times', 'dischargeDay', 'acute.renal.failure',
       'respiration', 'systolic.blood.pressure', 'diastolic.blood.pressure',
       'body.temperature', 'BMI', 'LVEF', 'weight', 'height',
       'creatinine.enzymatic.method', 'glomerular.filtration.rate',
       'hemoglobin', 'platelet', 'D.dimer', 'high.sensitivity.troponin',
       'brain.natriuretic.peptide', 'sodium', 'potassium', 'total.protein',
       'cholesterol', 'triglyceride']

scaler = StandardScaler()

# Fit and transform on training set only
x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])

print(" Numerical variables scaled successfully!")

##



