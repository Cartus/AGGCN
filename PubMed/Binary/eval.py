"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from data.loader import DataLoader, read_file
from model.trainer import GCNTrainer
from utils import torch_utils, nary_scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir']  + '/test.json'
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
data = read_file(data_file, vocab, opt, False)
batch = DataLoader(data, opt['batch_size'], opt, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v, k) for k, v in label2id.items()])

predictions = []
all_probs = []
cross_list = []
batch_iter = tqdm(batch)
for i, b in enumerate(batch_iter):
    cross_list += b[7]
    preds, probs, _ = trainer.predict(b)
    predictions += preds
    all_probs += probs

predictions = [id2label[p] for p in predictions]
print(predictions)
test_score, test_single_score = nary_scorer.score(batch.gold(), predictions, cross_list)
print("Test set evaluate result: cross {:.3f}, single {:.3f}".format(test_score, test_single_score))

print("Evaluation ended.")

