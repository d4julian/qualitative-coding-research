# %%
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import torch

codebook = pd.read_csv('updated_codebook.csv')
print(codebook.head())

blooms_df = codebook[['#', 'Bloom\'s', 'Description']].dropna()
blooms_df['Bloom\'s'] = blooms_df['Bloom\'s'] + ': ' + blooms_df['Description']
blooms_df['#'] = blooms_df['#'].astype(int)
blooms_df.sort_values(by='#', inplace=True)
bloom_dict = pd.Series(blooms_df['Bloom\'s'].values, index=blooms_df['#']).to_dict()
bloom_inverse_dict = blooms_df.set_index('Bloom\'s')['#'].to_dict()

print(bloom_inverse_dict)

competency_df = codebook[['Number', 'Competency']]
competency_dict = competency_df.set_index('Number')['Competency'].to_dict()
competency_inverse_dict = competency_df.set_index('Competency')['Number'].to_dict()
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

model_id = 'FacebookAI/roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_id)

encodings = tokenizer(
    list(data['combined_text']),
    truncation=True,
    padding=True,
    max_length=1024
)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

competency_dataset = Dataset(encodings, data['Competency_encoded'].tolist())
bloom_dataset = Dataset(encodings, data['Bloom\'s_encoded'].tolist())

def train_model(task_name, dataset, num_labels, output_dir, multi_label):
    model = RobertaForSequenceClassification.from_pretrained(model_id, num_labels=num_labels, problem_type="multi_label_classification" if multi_label else "single_label_classification")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        evaluation_strategy="no",
        logging_dir=f'./logs/{task_name}',
        logging_steps=10,
        save_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
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