import torch
from transformers import AutoModel
from transformers import BertTokenizerFast

@torch.no_grad()
def predict_rating(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    emb=torch.flatten(outputs.pooler_output)
    rating=str(torch.argmax(classifier(emb)).item()+1)
    return rating

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased')
model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased') #, return_dict=True
classifier=torch.nn.Sequential(
    torch.nn.Linear(in_features=768,out_features=5),
    torch.nn.Softmax(dim=-1)
)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.0001)
load_checkpoint("classifier.pth", classifier, optimizer, 0.001,)
text=""
print("Думаю человек написавший этот отзыв поставил бы "+predict_rating(text))
