import json
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Models import CNNModel
from model.eval.eval import f1_score

# Global variables
batch_size = 32
threshold = 0.5
save_after_iters = 10000
device = 'cuda:0'

def get_vocab(data_file):
	if os.path.isfile('../data/english_vocab.pkl'):
		vocab, word_to_ind = pickle.load(open('../data/english_vocab.pkl','rb'))
		return vocab, word_to_ind

	with open('../data/bioasq_dataset/allMeSH_2017.json', 'r', encoding="utf8", errors='ignore') as f:
		records = json.load(f)['articles']
	
	word_count = {}
	for record in records:
		words_in_abstract = record['abstractText'].lower().split(' ')
		for word in words_in_abstract:
			if word in word_count:
				word_count[word] += 1
			else:
				word_count[word] = 1

	vocab = [k for k in sorted(word_count, key=word_count.get, reverse=True)]
	# reduce the size of vocab to max size
	vocab = vocab[:vocab_size]
	# add the unknown token to the vocab
	vocab = ['$PAD$'] + vocab + ['unk']

	word_to_ind = {}
	for i, word in enumerate(vocab):
		word_to_ind[word] = i

	pickle.dump((vocab, word_to_ind), open('../data/english_vocab.pkl','wb'))

	return vocab, word_to_ind


def prepare_minibatch(data, mesh_to_idx, word_to_idx):
	X = []
	Y = []
	labels = []
	for article in data:
		word_seq = article['abstract'].lower().strip().split(' ')
		idx_seq = [word_to_idx[word] if word in word_to_idx else word_to_idx['unk'] for word in word_seq]
		idx_seq = idx[:max_seq_len]
		src_seq = np.zeros(max_seq_len)
		src_seq[0:len(idx_seq)] = idx_seq
		X.append(src_seq)
		tgt_seq = np.zeros(len(mesh_to_idx))
		tgt_idx = [mesh_to_idx[mesh] for mesh in article['mesh_labels']]
		tgt_seq[tgt_idx] = 1
		Y.append(tgt_idx)
		labels.append(article['mesh_labels'])
	X = np.vstack(X)
	Y = np.vstack(Y)
	return X, Y, labels


def train(model, criterion, mesh_to_idx, mesh_vocab, word_to_idx, src_vocab):
	# read the list of files to be used for training
	path = '../data/bioasq_dataset/train_data'
	list_files = os.listdir(path)

	iters = 0
	for file in list_files:
		file_content = json.load(open(path+'/'+file, 'r'))
		i = 0
		while i < len(file_content):
			input_idx_seq, target, _ = prepare_minibatch(file_content[i:i+batch_size], mesh_to_idx, word_to_idx)			
			input_idx_seq = input_idx_seq.to(device)
			target = target.to(device)
			predict, _ = model(input_idx_seq, input_mask)

			# computing the loss over the prediction
			loss = criterion(target, predict)
			print ("loss: ", loss)

			# back-propagation
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(transformer.parameters(), clip_norm)
			optimizer.step()

			i += batch_size
			iters += 1

			if iters % save_after_iters == 0:
				torch.save(model.state_dict(), '../saved_models/model.pt')

	return model 


def validate(model, mesh_to_idx, mesh_vocab, word_to_idx, src_vocab):
	path = '../data/bioasq_dataset/val_data'
	list_files = os.listdir(path)
	
	true_labels = []
	pred_labels = []
	for file in list_files:
		file_content = json.load(open(path+'/'+file, 'r'))
		i = 0
		while i < len(file_content):
			input_idx_seq, target, true_labels_batch = prepare_minibatch(file_content[i:i+batch_size], mesh_to_idx, word_to_idx)			
			input_idx_seq = input_idx_seq.to(device)
			predict, _ = model(input_idx_seq, input_mask)
			predict[predict>threshold] = 1
			predict[predict<threshold] = 0
			predict = predict.to('cpu').numpy()

			for j in predict.shape[0]:
				nnz_idx = np.nonzero(predict[j, :])[0]
				pred_labels_article = [mesh_vocab[idx] for idx in nnz_idx]
				pred_labels.append(pred_labels_article)

			true_labels.extend(true_labels_batch)
			i += batch_size

	f1_score_micro, f1_score_macro = f1_score(true_labels, pred_labels)
	print ("f1 score micro: ", f1_score_micro, " f1 score macro: ", f1_score_macro)

	return f1_score_micro

if __name__ == '__main__':

	# create the vocabulary for the input 
	src_vocab, word_to_idx = get_vocab(data_file)
	print("vocabulary of size: ", len(src_vocab))

	# create the vocabulary of mesh terms
	with open('../data/mesh_to_idx.pkl', 'rb') as fread:
		mesh_to_idx = pickle.load(fread)
	
	mesh_vocab = [" "] * len(mesh_to_idx)
	for mesh, idx in mesh_to_idx.items():
		mesh_vocab[idx] = mesh


	# read source word embedding matrix
	emb_mat_src = read_embeddings('../data/embeddings/word.processed.embeddings', src_vocab)
	emb_mat_src = torch.tensor(emb_mat_src).to(out_device)


	loss_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
	optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, betas=(0.9,0.999))
	# optimizer = torch.optim.SGD(transformer.parameters(), lr=learning_rate)

	# setting different model parameters
	n_src_vocab = len(src_vocab)
	n_tgt_vocab = len(mesh_to_idx)
	max_seq_len = 1000
	d_word_vec = 200
	dropout = 0.1
	

	model = CNNModel(n_src_vocab, n_tgt_vocab, max_len_seq, d_word_vec, emb_mat_src, dropout=dropout)
	model = nn.DataParallel(model, output_device=device)
	model.to(device)

	train(model, criterion, mesh_to_idx, mesh_vocab, word_to_idx, src_vocab)







