
import torch
from transformers import AutoTokenizer, LlamaForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"


model = LlamaForSequenceClassification.from_pretrained("./llama3").to(device)


def get_embed(batch):
    with torch.no_grad():
        op = model(batch, output_hidden_states=True)
        embeds = torch.mean(op.hidden_states[-1], dim=1)

        return embeds
