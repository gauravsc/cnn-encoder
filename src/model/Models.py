import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNModel(nn.Module):
	def __init__(self, n_src_vocab, n_tgt_vocab, len_max_seq, d_word_vec, emb_mat_src, dropout=0.1):
		super().__init__()
		n_src_vocab, d_word_vec = emb_mat_src.size()
		self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=0)
		self.src_word_emb.load_state_dict({'weight': emb_mat_src})
		# self.conv_layer_1 = nn.Conv1d(d_word_vec, 1024, 3, padding=1, stride=1)

		self.conv_layer_1_1 = nn.Conv1d(d_word_vec, 256, 1, padding=0, stride=1)
		self.conv_layer_1_3 = nn.Conv1d(d_word_vec, 256, 3, padding=1, stride=1)
		self.conv_layer_1_5 = nn.Conv1d(d_word_vec, 256, 5, padding=2, stride=1)

		self.conv_layer_2 = nn.Conv1d(768, 512, 3, padding=1, stride=1)
		self.conv_layer_3 = nn.Conv1d(512, 256, 3, padding=1, stride=1)
		self.conv_layer_4 = nn.Conv1d(256, 256, 3, padding=1, stride=1)
		self.maxpool = nn.MaxPool1d(2)
		self.dropout = nn.Dropout(p=dropout)
		self.fc_layer_1 = nn.Linear((len_max_seq//16)*256, 256)
		self.output_layer = nn.Linear(256, n_tgt_vocab)
		self.relu_activation = nn.ReLU()


	def forward(self, input_idxs):
		output = self.src_word_emb(input_idxs)
		output = output.permute(0,2,1)

		# output = self.conv_layer_1(output)
		# output = self.relu_activation(output)
		# output = self.maxpool(output)

		output_1 = self.conv_layer_1_1(output)
		output_1 = self.relu_activation(output_1)
		output_1 = self.maxpool(output_1)

		output_3 = self.conv_layer_1_3(output)
		output_3 = self.relu_activation(output_3)
		output_3 = self.maxpool(output_3)

		output_5 = self.conv_layer_1_5(output)
		output_5 = self.relu_activation(output_5)
		output_5 = self.maxpool(output_5)

		output = torch.cat((output_1, output_3, output_5), 1)

		output = self.conv_layer_2(output)
		output = self.relu_activation(output)
		output = self.maxpool(output)
		
		output = self.conv_layer_3(output)
		output = self.relu_activation(output)
		output = self.maxpool(output)

		output = self.conv_layer_4(output)
		output = self.relu_activation(output)
		output = self.maxpool(output)

		output = self.dropout(output.view(output.size()[0], -1))
		output_fc_layer = self.relu_activation(self.fc_layer_1(output))
		target = self.output_layer(self.dropout(output_fc_layer))

		return target, output_fc_layer
