import torch
import pandas as pd
from transformers import AutoModel
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased')
model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased') #, return_dict=True
model=model.to('cuda')

@torch.no_grad()
def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt').to("cuda")
    outputs = model(**inputs)
    return torch.flatten(outputs.pooler_output)

data=pd.read_csv('geo-reviews-dataset-2023.tskv',sep=';')
data.drop(data.columns[[0,1,3]], axis=1, inplace=True)
data.columns=['rating','text']
data=data.dropna()
data['rating']=data['rating'].replace(to_replace='\D', value='',regex=True)
data['text']=data['text'].replace(to_replace='text=', value='',regex=True)
data['text']=data['text'].replace(to_replace='[^ёйцукенгшщзхъфывапролджэячсмитьбюЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ., ]', value='',regex=True)

data1=data.loc[data['rating']==1].iloc[:12071]
data2=data.loc[data['rating']==2].iloc[:12071]
data3=data.loc[data['rating']==3].iloc[:12071]
data4=data.loc[data['rating']==4].iloc[:12071]
data5=data.loc[data['rating']==5].iloc[:12071]
data=pd.concat([data1,data2,data3,data4,data5])

for i in range(768):
    data[str(i)]=[0]*data.shape[0]

for i in range(data.shape[0]):
    print(i)
    emb=predict(data.iloc[i, 1])
    for j in range(768):
        data.iloc[i, j+2]=float(emb[j])

data.drop(data.columns[[1]], axis=1, inplace=True)
data.to_csv('dataset.csv',index=False,encoding='utf-8',sep=';')