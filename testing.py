##############################################################################################
# Prepare dataset for K-Fold
##############################################################################################
array = dataset_fs.values
X_fs = dataset_fs.drop('diagnosis', axis=1)
y_fs = dataset_fs['diagnosis']

print('Class labels:', np.unique(y_fs))

# Standardize the features
scaler = StandardScaler()
X_fs_scaled = scaler.fit_transform(X_fs)

# Set up Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

accuracy_scores = []

for train_index, validation_index in kf.split(X_fs_scaled, y_fs):
    # From the training indices, randomly select 50% for actual training
    X_train_kfold, _, y_train_kfold, _ = train_test_split(X_fs_scaled[train_index], y_fs.iloc[train_index], train_size=0.5, random_state=None, stratify=y_fs.iloc[train_index])
    
    X_validation_fs = X_fs_scaled[validation_index]
    y_validation_fs = y_fs.iloc[validation_index]
    
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_kfold, y_train_kfold)
    
    y_pred = logistic_model.predict(X_validation_fs)
    
    accuracy = accuracy_score(y_validation_fs, y_pred)
    accuracy_scores.append(accuracy)
    
    print(f"Fold Accuracy = {accuracy:.4f}")

# Calculate and print the average accuracy across all folds
average_accuracy = np.mean(accuracy_scores)
print(f"\nAverage accuracy over {kf.get_n_splits()} folds: {average_accuracy:.4f}")