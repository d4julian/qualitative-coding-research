# %%
import torch
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm

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
# %%

competency_model_path = './trained_models/competency_model'
bloom_model_path = 'google/bigbird-roberta-base'

competency_tokenizer = BigBirdTokenizer.from_pretrained(competency_model_path)
bloom_tokenizer = BigBirdTokenizer.from_pretrained(bloom_model_path)

competency_model = BigBirdForSequenceClassification.from_pretrained(competency_model_path)
bloom_model = BigBirdForSequenceClassification.from_pretrained(bloom_model_path)
competency_model.eval()
bloom_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
competency_model.to(device)
bloom_model.to(device)

competency_pipeline = pipeline(
    "text-classification",
    model=competency_model_path,
    tokenizer=competency_model_path,
    top_k=None,
    device=device,
    max_length=4096,
    truncation=True
)

bloom_pipeline = pipeline(
    "text-classification",
    model=bloom_model_path,
    tokenizer=bloom_model_path,
    top_k=1,
    device=device,
    max_length=4096,
    truncation=True
)

def classify_competency(text, threshold=0.6):
    results = competency_pipeline(text)
    all_scores = results[0]
    
    predicted_labels = [item['label'] for item in all_scores if item['score'] >= threshold]
    predicted_with_scores = [(item['label'], item['score']) for item in all_scores if item['score'] >= threshold]
    
    return predicted_labels, predicted_with_scores

def classify_bloom(text):
    result = bloom_pipeline(text)
    all_scores = result[0]
    predicted_label = all_scores[0]['label']
    score = all_scores[0]['score']
    return predicted_label, score

test_code = """
Code: def numberGame(nums):
        sorted_array = nums.sort()
        for i in range(0,len(nums), 2):
            temp = sorted_array[i]
            sorted_array[i] = sorted_array[i+1]
            sorted_array[i+1] = temp
        return sorted_array
"""

competency_pred, competency_pred_scores = classify_competency(test_code)
print("**Competency Classification**")
print("Predicted Labels:", competency_pred)
print("Predicted Labels with Scores:", competency_pred_scores)

bloom_pred, bloom_score = classify_bloom(test_code)
print("\n**Bloom Classification**")
print("Predicted Label:", bloom_pred)
print("Score:", bloom_score)

# %%

remaining_data = pd.read_csv('remaining_data.csv')
remaining_data['code'] = remaining_data['code'].fillna('')
remaining_data['error'] = remaining_data['error'].fillna('')
remaining_data['issue'] = remaining_data['issue'].fillna('')
remaining_data['combined_text'] = (
    "Code: " + remaining_data['code'] + "\n" +
    "Error: " + remaining_data['error'] + "\n" +
    "Issue: " + remaining_data['issue']
).replace('NA', '')
#%% 

for index, row in tqdm(remaining_data.iterrows()):
    competency_pred, competency_pred_scores = classify_competency(row['combined_text'])
    bloom_pred, bloom_score = classify_bloom(row['combined_text'])
    remaining_data.at[index, 'Competency_num'] = ','.join(
        str(competency_inverse_dict.get(label, '')) for label in competency_pred
    )
    remaining_data.at[index, 'Learning_num'] = int(bloom_inverse_dict[bloom_pred])

remaining_data.to_csv('remaining_data_output.csv', index=False)