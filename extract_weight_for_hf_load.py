from src.contriever import Contriever
from transformers import AutoTokenizer, BertConfig, BertModel
import torch

train_strategy = "inbatch"
path = 'checkpoint/tmp/contriever_inbatch/checkpoint/step-500/checkpoint.pth'
model = torch.load(path)
model = model['model']

# iterate model keys and get only encoder_q keys
encoder = {}
if train_strategy == "moco":
    for key in model.keys():
        # in MoCo, we only use encoder_q
        if 'encoder_q' in key:
            # remove encoder_q. from key
            encoder[key.removeprefix("encoder_q.")] = model[key]
elif train_strategy == "inbatch":
    for key in model.keys():
        if 'encoder' in key:
            # remove encoder. from key
            encoder[key.removeprefix("encoder.")] = model[key]

config = BertConfig.from_pretrained("facebook/contriever")
contriever = BertModel(config)

# Load the weights from the pre-trained model we set strict to False to ignore the pooler layer
contriever.load_state_dict(encoder, strict=False)

# Check if the partial weights are equal
assert torch.all(contriever.embeddings.word_embeddings.weight.detach().cpu() == encoder['embeddings.word_embeddings.weight'].cpu()), "weights are not equal"
