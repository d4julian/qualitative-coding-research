# %%
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import torch

codebook = pd.read_csv('updated_codebook.csv')

blooms_df = codebook[['#', 'Bloom\'s', 'Description']].dropna()
blooms_df['Bloom\'s'] = blooms_df['Bloom\'s'] + ': ' + blooms_df['Description']
blooms_df['#'] = blooms_df['#'].astype(int)
blooms_df.sort_values(by='#', inplace=True)
blooms_df['zero_index'] = blooms_df['#'] - 1
bloom_dict = pd.Series(blooms_df['Bloom\'s'].values, index=blooms_df['#']).to_dict()
zero_index_bloom_dict = pd.Series(blooms_df['Bloom\'s'].values, index=blooms_df['zero_index']).to_dict()
print(zero_index_bloom_dict)
bloom_inverse_dict = blooms_df.set_index('Bloom\'s')['zero_index'].to_dict()
print(bloom_inverse_dict)

competency_df = codebook[['Number', 'Competency']]
competency_df['zero_index'] = competency_df['Number'] - 1
competency_dict = competency_df.set_index('Number')['Competency'].to_dict()
zero_index_competency_dict = competency_df.set_index('zero_index')['Competency'].to_dict()
print(zero_index_competency_dict)
competency_inverse_dict = competency_df.set_index('Competency')['zero_index'].to_dict()
print(competency_inverse_dict)
# %%

data = pd.read_csv('filled_train_data.csv')

data['code'] = data['code'].fillna('')
data['error'] = data['error'].fillna('')
data['issue'] = data['issue'].fillna('')
data['combined_text'] = (
    "Code: " + data['code'] + "\n" +
    "Error: " + data['error'] + "\n" +
    "Issue: " + data['issue']
)

competency_encoder, bloom_encoder = MultiLabelBinarizer(), LabelEncoder()

competency_encoder.fit(competency_df['Competency'])
bloom_encoder.fit(blooms_df['Bloom\'s'])

data['Competency_List'] = data['Competency - Num'].astype(str).str.split(',').apply(
    lambda nums: [competency_dict[int(num.strip())] for num in nums]
)
data['Competency_encoded'] = list(competency_encoder.fit_transform(data['Competency_List']))
data['Bloom\'s'] = data['Learning - Num'].astype(int).map(bloom_dict)
data['Bloom\'s_encoded'] = bloom_encoder.transform(data['Bloom\'s'])

print(data.head())
# %%

model_id = 'google/bigbird-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

encodings = tokenizer(
    list(data['combined_text']),
    truncation=True,
    padding=True,
    max_length=4096
)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, multi_label):
        self.encodings = encodings
        self.labels = labels
        self.multi_label = multi_label
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float if self.multi_label else torch.long)
        return item
    def __len__(self):
        return len(self.labels)

competency_dataset = Dataset(encodings, data['Competency_encoded'].tolist(), multi_label=True)
bloom_dataset = Dataset(encodings, data['Bloom\'s_encoded'].tolist(), multi_label=False)

def train_model(task_name, dataset, num_labels, output_dir, multi_label):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=num_labels, 
        #problem_type="multi_label_classification" if multi_label else "single_label_classification",
        id2label=zero_index_competency_dict if multi_label else zero_index_bloom_dict,
        label2id=competency_inverse_dict if multi_label else bloom_inverse_dict

    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        learning_rate=1e-5,
        eval_strategy="no",
        logging_dir=f'./logs/{task_name}',
        logging_steps=1,
        save_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )
    
    trainer.train()
    
    trainer.save_model(f'./trained_models/{task_name}_model')
    tokenizer.save_pretrained(f'./trained_models/{task_name}_model')
    
    print(f"Training for {task_name} completed and model saved to './trained_models/{task_name}_model'")

train_model(
    task_name='competency',
    dataset=competency_dataset,
    num_labels=len(competency_inverse_dict),
    output_dir='./results_competency',
    multi_label=True
)

train_model(
    task_name='bloom',
    dataset=bloom_dataset,
    num_labels=len(bloom_inverse_dict),
    output_dir='./results_bloom',
    multi_label=False
)
# %%