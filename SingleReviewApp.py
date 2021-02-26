from Model.model import *


class SentimentAnalysisApp:

	def __init__(self, tokenizerPath, modelPath):

		self.classifier = loadModel(tokenizerPath, modelPath)
		self.singleReviewPage()

	def singleReviewPage(self):

		# Page Title
		st.title('Employee Sentiment Analysis')

		# Page Sub Title
		st.subheader('Single Review')

		# Input from user
		sentence = st.text_area("Input employee review below", "Great working environment!")

		# Submit button
		submit = st.button("Get Sentiment")

		# Display prediction
		if submit:

			# start time
			start_time = time.time()	

			with st.spinner('Predicting...'):
				self.displayPrediction(sentence)

			# end time
			end_time = time.time()

			# get time taken to finish predicting
			prediction_time = end_time - start_time

			# display time taken
			st.success('Prediction was successfully completed in {:.2f} s'.format(prediction_time))


	def displayPrediction(self, sentence):

		# Run predictions
		label, prob = self.classifier.predict(sentence)

		if label == "Positive":
			st.success("{} Sentiment: {}".format(label, prob))

		elif label == "Neutral":
			st.warning("{} Sentiment: {}".format(label, prob))

		else:
			st.error("{} Sentiment: {}".format(label, prob))


if __name__ == '__main__':

	tokenizerPath = 'roberta-base'
	modelPath = 'Model/best-models-0.8137.pt'

	SentimentAnalysisApp(tokenizerPath, modelPath)