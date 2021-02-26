# Import the necessary libaries
import time
import torch
import base64
import pandas as pd
import torch.nn as nn
import streamlit as st 
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel


class SentimentClassifier(nn.Module):

	def __init__(self, backbone, num_classes=3):
		super().__init__()
		self.backbone = backbone
		self.classifier = nn.Sequential(
		    nn.Dropout(0.5),
		    nn.Linear(in_features=768, out_features=num_classes),
		    nn.BatchNorm1d(num_classes),
		)

	def forward(self, input, attention):
		_, cls_hs = self.backbone(input, attention_mask=attention, return_dict=False)
		return self.classifier(cls_hs)


class Pipeline:

	def __init__(self, tokenizerPath, modelPath, batch_size=8):

		# Set device
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Initialise tokenizer
		self.tokenizer = RobertaTokenizer.from_pretrained(tokenizerPath)

		# Initialise model
		backbone = RobertaModel.from_pretrained(tokenizerPath)
		self.model = SentimentClassifier(backbone).to(self.device)

		# Load saved model weights
		self.model.load_state_dict(torch.load(modelPath, map_location=self.device))
		
		# Labels mapping
		self.mapping = {0: 'Positive',
						1: 'Neutral',
						2: 'Negative'}

		# Batch size
		self.batch_size = batch_size

	def tokenize(self, sentence):

		# Convert to list
		sentences = [sentence] if isinstance(sentence, str) else sentence.Comments.tolist()

		# Tokenize and convert to tensors
		seq, mask = self.tokenizer.batch_encode_plus(
			sentences,
			max_length=35,
			padding='max_length',
			truncation=True,
			return_tensors='pt').values()

		# Load tensors into TensorDataset
		dataset = TensorDataset(seq, mask)

		# Load dataset into dataloader
		dataloader = DataLoader(dataset, batch_size=self.batch_size)

		return dataloader

	def predict(self, sentence):

		# Initialise an empty list to collect the predictions
		predictions = []
		
		# Set model to evaluation mode
		self.model.eval()

		# Tokenize and load sentences into dataloader
		dataloader = self.tokenize(sentence)

		with torch.no_grad():

			# Loop through dataloader
			for seqs, masks in dataloader:

				# Move seqs and masks to device
				seqs, masks = seqs.to(self.device), masks.to(self.device)

				# Get model predictions
				logits = F.softmax(self.model(seqs, masks), dim=1)
				probs, preds = torch.max(logits, dim=1)

				# Append preds and probs to predictions
				for pred, prob in zip(preds, probs):
					predictions.append((self.mapping[pred.item()], round(prob.item(), 2)))

		if isinstance(sentence, str):
			label, prob = predictions[0]
			return label, prob

		# Return predictions dataframe
		return sentence.merge(pd.DataFrame(predictions, columns=["Sentiment", "Confidence_score"]), left_index=True, right_index=True)


@st.cache(allow_output_mutation=True)
def loadModel(tokenizerPath, modelPath):
	classifier = Pipeline(tokenizerPath, modelPath)
	return classifier