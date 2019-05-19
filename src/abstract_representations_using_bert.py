import json
import os
import pickle
import csv
import numpy as np
import random as rd
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
from model.Models import BERTCLassifierModel
from eval.eval import f1_score
from utils.embedding_operations import read_embeddings


# Global variables
batch_size = 4
clip_norm = 10.0
max_epochs = 100
device = 'cuda:0'
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# set random seed
rd.seed(9001)
np.random.seed(9001)

def prepare_minibatch(data, mesh_to_idx, tokenizer):
	X = []
	Y = []
	input_mask = []
	labels = []
	for article in data:
		# word_seq = article['abstract'].lower().strip().split(' ')
		tokenized_text = tokenizer.tokenize('[CLS] '+article['abstract'].lower())[0:512]
		idx_seq = tokenizer.convert_tokens_to_ids(tokenized_text)
		src_seq = np.zeros(max_seq_len)
		src_seq[0:len(idx_seq)] = idx_seq
		X.append(src_seq)
		mask = np.zeros(max_seq_len)
		mask[0:len(idx_seq)] = 1
		input_mask.append(mask)
		tgt_seq = np.zeros(len(mesh_to_idx))
		tgt_idx = [mesh_to_idx[mesh] for mesh in article['mesh_labels']]
		tgt_seq[tgt_idx] = 1
		Y.append(tgt_seq)
		labels.append(article['mesh_labels'])
	X = np.vstack(X)
	Y = np.vstack(Y)
	input_mask = np.vstack(input_mask)
	return X, input_mask, Y, labels



def validate(model, mesh_to_idx, mesh_vocab, tokenizer, threshold):
	model = model.eval()
	path = '../data/bioasq_dataset/val_data'
	list_files = os.listdir(path)
	print (list_files)

	true_labels = []; pred_labels = []
	for file in list_files:
		file_content = json.load(open(path+'/'+file, 'r'))
		i = 0
		while i < len(file_content):
			input_idx_seq, input_mask, target, true_labels_batch = prepare_minibatch(file_content[i:i+4], mesh_to_idx, tokenizer)			
			input_idx_seq = torch.tensor(input_idx_seq).to(device, dtype=torch.long)
			input_mask = torch.tensor(input_mask).to(device, dtype=torch.long)
			predict = model(input_idx_seq, input_mask)
			predict = F.sigmoid(predict)
			predict[predict>threshold] = 1
			predict[predict<threshold] = 0
			predict = predict.data.to('cpu').numpy()

			for j in range(predict.shape[0]):
				nnz_idx = np.nonzero(predict[j, :])[0]
				pred_labels_article = [mesh_vocab[idx] for idx in nnz_idx]
				pred_labels.append(pred_labels_article)

			true_labels.extend(true_labels_batch)
			i += 4

	# for k in range(len(true_labels)):
	# 	print (true_labels[k])
	# 	print (pred_labels[k])

	f1_score_micro, f1_score_macro = f1_score(true_labels, pred_labels) 
	print ("f1 score micro: ", f1_score_micro, " f1 score macro: ", f1_score_macro)

	return f1_score_micro


def extract_vector_representations(model, data, tokenizer):
	model = model.eval()
	# fieldnames = ['_id', 'table', 'abstract']
	X = []; Mask = []; 
	for rec in data:
		text = rec['abstract']
		tokenized_text = tokenizer.tokenize('[CLS] ' + text.lower())[0:512]
		idx_seq = tokenizer.convert_tokens_to_ids(tokenized_text)
		src_seq = np.zeros(max_seq_len)
		src_seq[0:len(idx_seq)] = idx_seq
		X.append(src_seq)
		mask = np.zeros(max_seq_len)
		mask[0:len(idx_seq)] = 1
		Mask.append(mask)

	X = np.vstack(X)
	Mask = np.vstack(Mask)

	X = torch.tensor(X).to(device, dtype=torch.long)
	Mask = torch.tensor(Mask).to(device, dtype=torch.long)

	_, encoder_output = model(X, Mask)
	encoder_output = encoder_output.data.to('cpu').numpy()

	res = {}; results = []
	for idx, rec in enumerate(data):
		res['table'] = rec['table']
		res['_id'] = rec['_id']
		res['representation'] = encoder_output[idx, :].tolist()
		results.append(res)
	return results


if __name__ == '__main__':
	# create  csv dictreader object
	csv_reader = csv.DictReader(open('../data/all_text_abstracts_mapped_to_table_and_id.csv','r'))

	# Load pre-trained model tokenizer (vocabulary)
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)

	# create the vocabulary of mesh terms
	with open('../data/mesh_to_idx.pkl', 'rb') as fread:
		mesh_to_idx = pickle.load(fread)
	
	mesh_vocab = [" "] * len(mesh_to_idx)
	for mesh, idx in mesh_to_idx.items():
		mesh_vocab[idx] = mesh

	# setting different model parameters
	n_tgt_vocab = len(mesh_to_idx)
	max_seq_len = 512
	d_word_vec = 200
	dropout = 0.1
	learning_rate = 0.005

	print ("Starting to load the saved model .....")
	model = BERTCLassifierModel(n_tgt_vocab, dropout=dropout)
	model.load_state_dict(torch.load('../saved_models/bert_based/model.pt'))
	model.to(device)
	print ("Done loading the saved model .....")

	results = []; data = []; i = 0; ctr = 0;
	for i, row in enumerate(csv_reader):
		print (i)
		data.append(row)
		if (i+1) % batch_size == 0:
			results.extend(extract_vector_representations(model, data, tokenizer))
			data = []
			if len(results) % (4*1250) == 0:
				print ("Dumping :", len(results), " results into the file on disk")
				json.dump(results, open('../data/vector_representations/vector_representations_'+str(ctr)+'.json', 'w'))
				results = []; ctr += 1

	if len(results) != 0:
		print ("Dumping :", len(results), " results into the file on disk")
		json.dump(results, open('../data/vector_representations/vector_representations_'+str(ctr)+'.json', 'w'))


