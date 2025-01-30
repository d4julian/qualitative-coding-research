import pandas as pd
from dataclasses import dataclass
import spacy

import numpy as np
import re
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from transformers import pipeline

codebook = pd.read_csv('codebook.csv')
print(codebook.head())

blooms_df = codebook[['#', 'Bloom\'s', 'Description']].dropna()
blooms_df.sort_values(by='#', inplace=True)
print(blooms_df)
#bloom_dict = pd.Series(blooms_df['Bloom\'s'].values, index=blooms_df['#']).to_dict()
bloom_inverse_dict = blooms_df.set_index('Bloom\'s')['#'].to_dict()
#print(bloom_dict)

competency_df = codebook[['Number', 'Competency', 'Python Content', 'C++ Content']]
competency_dict = competency_df.set_index('Number')['Competency'].to_dict()
competency_inverse_dict = competency_df.set_index('Competency')['Number'].to_dict()
print(competency_df)

train_data = pd.read_csv('train_data.csv')
train_data = train_data[['context_name', 'code', 'error', 'issue', 'response_text', 'Competency - Num', 'Competency - Text', 'Learning - Num', 'Learning - Text']]
train_data['combined_text'] = train_data[['code', 'error', 'issue']].astype(str).agg(' '.join, axis=1).str.lower()

candidate_labels = competency_df['Competency'].dropna().unique().tolist()
bloom_labels = blooms_df['Bloom\'s'].dropna().unique().tolist()

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
print("\nZero-Shot Classification Pipeline Initialized.")

def assign_competencies(text, candidate_labels, threshold=0.8, top_k=4):
    result = classifier(text, candidate_labels, multi_label=True, batch_size=128)
    # Filter by threshold first, then sort and get top k
    filtered_pairs = [(label, score) for label, score in zip(result['labels'], result['scores']) if score >= threshold]
    sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)[:top_k]
    assigned = [
        competency_inverse_dict[label]
        for label, score in sorted_pairs
    ]
    return assigned

def assign_blooms(text, bloom_labels):
    result = classifier(text, bloom_labels, multi_label=False)
    return bloom_inverse_dict[result['labels'][0]]

train_data['Competency - Num'] = train_data['combined_text'].apply(lambda x: assign_competencies(x, candidate_labels))
train_data['Learning - Num'] = train_data['combined_text'].apply(lambda x: assign_blooms(x, bloom_labels))

train_data.drop(columns=['combined_text'], inplace=True)
train_data.to_csv('output.csv', index=False)