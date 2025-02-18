import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score
import numpy as np

codebook = pd.read_csv('updated_codebook.csv')

blooms_df = codebook[['#', 'Bloom\'s', 'Description']].dropna()
blooms_df['Bloom\'s'] = blooms_df['Bloom\'s'] + ': ' + blooms_df['Description']
blooms_df['#'] = blooms_df['#'].astype(int)
blooms_df.sort_values(by='#', inplace=True)
bloom_dict = pd.Series(blooms_df['Bloom\'s'].values, index=blooms_df['#']).to_dict()
bloom_inverse_dict = blooms_df.set_index('Bloom\'s')['#'].to_dict()

competency_df = codebook[['Number', 'Competency']]
competency_dict = competency_df.set_index('Number')['Competency'].to_dict()
competency_inverse_dict = competency_df.set_index('Competency')['Number'].to_dict()

data = pd.read_csv('filled_train_data.csv')

x = data['combined_text'] = data[['code', 'error', 'issue']].astype(str).agg(' '.join, axis=1).replace('NA', '')
y_competency = data['Competency - Num'].astype(str).str.split(',').apply(lambda x: [competency_dict[int(i.strip())] for i in x])
y_bloom = data['Learning - Num'].astype(int).apply(lambda x: bloom_dict[x])

bloom_model = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), LogisticRegression(solver='liblinear', class_weight='balanced'))
competency_model = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced')))

mlb = MultiLabelBinarizer()
y_competency_bin = mlb.fit_transform(y_competency)

bloom_model.fit(x, y_bloom)
competency_model.fit(x, y_competency_bin)

test_code = """
Code: def numberGame(nums):
        sorted_array = nums.sort()
        for i in range(0,len(nums), 2):
            temp = sorted_array[i]
            sorted_array[i] = sorted_array[i+1]
            sorted_array[i+1] = temp
        return sorted_array
"""

bloom_predict = bloom_model.predict([test_code])
competency_predict = competency_model.predict([test_code])
print(f"Bloom's Prediction: {bloom_predict}")
print(f"Competency Prediction: {mlb.inverse_transform(competency_predict)}")

remaining_data = pd.read_csv('remaining_data.csv')
remaining_data['combined_text'] = remaining_data[['code', 'error', 'issue']].astype(str).agg(' '.join, axis=1).replace('NA', '')
remaining_data['Competency_num'] = remaining_data['Competency_num'].astype(str)

for index, row in remaining_data.iterrows():
    bloom_pred = bloom_model.predict([row['combined_text']])[0]
    
    # **Updated Prediction Handling**
    competency_pred = competency_model.predict([row['combined_text']])  # Removed [0]
    competency_labels = mlb.inverse_transform(competency_pred)[0]       # Pass the 2D array directly
    
    remaining_data.at[index, 'Competency_num'] = ','.join(
        str(competency_inverse_dict.get(label, '')) for label in competency_labels
    )
    remaining_data.at[index, 'Learning_num'] = int(bloom_inverse_dict.get(bloom_pred, 0))  # Added get with default

remaining_data.to_csv('remaining_data_output.csv', index=False)