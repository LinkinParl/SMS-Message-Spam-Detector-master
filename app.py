from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

# Our 1st interface
@app.route('/') # To pecify URL that should trigger the execution of Home
def home():
	return render_template('home.html')

# Our 2nd Interface
@app.route('/predict',methods=['POST']) 
# Accessing text_data, pre-proces it & make predictions, then store the model
def predict():
	df= pd.read_csv("text_data.csv", encoding="latin-1")
	
	# Features and Labels
	df['label'] = df['Label'].map({'Non-Spam': 0, 'Spam': 1})
	X = df['Message_body']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)

	# Frontend
	# POST -> Transport the form data to server
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)