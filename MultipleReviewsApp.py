from Model.model import *


class SentimentAnalysisApp:

	def __init__(self, tokenizerPath, modelPath):

		self.classifier = loadModel(tokenizerPath, modelPath)
		self.multipleReviewsPage()

	def multipleReviewsPage(self):

		# Page Title
		st.title('Employee Sentiment Analysis')

		# Page Sub Title
		st.subheader('Multiple Reviews')

		# Input from user
		fileUpload = st.file_uploader("Upload Employee Reviews CSV File Here", type=['csv'])

		# Submit button
		submit = st.button("Get Sentiment(s)")

		# Get predictions
		if fileUpload and submit:

			try:
				# Read comments into a dataframe
				comments = pd.read_csv(fileUpload, usecols=["Comments"])

				# start time
				start_time = time.time()

				with st.spinner('Predicting...'):
					# Run predictions
					predictions = self.classifier.predict(comments)

				# end time
				end_time = time.time()

				# get time taken to finish predicting
				prediction_time = end_time - start_time

				# display time taken
				st.success('Prediction(s) were successfully completed in {:.2f} s'.format(prediction_time))

				# Provide download link to predictions
				st.markdown(self.downloadPredictions(predictions, fileUpload.name), unsafe_allow_html=True)

			except MemoryError:

				# Display error message
				st.error("Out of memory! Please reduce the number of comments in the uploaded file and try again!")

			except:

				# Display error message
				st.error("Please ensure that the uploaded file has the following format!")

				# Show sample dataframe
				st.dataframe(pd.DataFrame({"Comments":['sample review 1', 'sample review 2']}))


	@staticmethod
	def downloadPredictions(dataframe, fileName):

		# Set file name
		fileName = fileName.replace('.csv', '_predictions.csv') 

		# Convert predictions dataframe to csv
		predictions_csv = dataframe.to_csv(index=False)

		# Encode the csv file
		b64 = base64.b64encode(predictions_csv.encode()).decode()

		# Return html 
		return f'<a href="data:file/csv;base64,{b64}" download="{fileName}">Click Here To Download The Sentiments</a>'


if __name__ == '__main__':

	tokenizerPath = 'roberta-base'
	modelPath = 'Model/best-models-0.8137.pt'

	SentimentAnalysisApp(tokenizerPath, modelPath)