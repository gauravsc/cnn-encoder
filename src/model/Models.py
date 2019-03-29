import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNModel(nn.Module):
	 def __init__(
			self,
			n_src_vocab, n_tgt_vocab, len_max_seq, d_word_vec, emb_mat_src,
			d_model, dropout=0.1):

		super().__init__()

		n_src_vocab, d_word_vec = emb_mat_src.size()
		self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
		self.src_word_emb.load_state_dict({'weight': emb_mat_src})

		self.conv_layer_1 = nn.Conv1d(d_word_vec, 256, 3, padding=1, stride=1)
		self.conv_layer_2 = nn.Conv1d(256, 256, 3, padding=1, stride=1)
		self.conv_layer_3 = nn.Conv1d(256, 256, 3, padding=1, stride=1)
		self.conv_layer_4 = nn.Conv1d(256, 64, 3, padding=1, stride=1)
		self.maxpool = nn.MaxPool1d(3)
		self.fc_layer_1 = nn.Linear(len_max_seq//81, 256)
		self.output_layer = nn.Linear(256, n_tgt_vocab)
		self.relu_activation = nn.ReLU()


	def forward(input_idxs, input_masks):

		output = self.src_word_emb(input_idxs)
		output = self.conv_layer_1(output)
		output = self.maxpool(output)
		output = self.conv_layer_2(output)
		output = self.relu_activation(output)
		output = self.maxpool(output)
		output = self.conv_layer_3(output)
		output = self.relu_activation(output)
		output = self.maxpool(output)
		output = self.conv_layer_4(output)
		output = self.relu_activation(output)
		output = self.maxpool(output)
		output = output.view(output.size()[0], -1)
		output = self.fc_layer_1(output)
		output = self.output_layer(output)

		return output
