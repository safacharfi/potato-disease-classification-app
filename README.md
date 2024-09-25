# Potato Disease Prediction

This project aims to predict potato plant diseases using a Convolutional Neural Network (CNN) model. It includes a React-based web interface for capturing leaf images and displaying predictions, with a backend powered by FastAPI and TensorFlow Serving.
##Project Workflow
![image](https://github.com/user-attachments/assets/0a5d5f8a-4b06-4545-9ac2-696c9b381c93)

## Project Structure

The project consists of several components:

1. Data Processing:
   - Data Cleaning & Preprocessing
   - Data Augmentation

2. Model:
   - CNN (Convolutional Neural Network)
   - Trained on a TensorFlow dataset

3. Backend:
   - TensorFlow Serving for model deployment
   - FastAPI for API endpoints

4. Frontend:
   - React.js web application

## Technologies Used

- TensorFlow: For dataset handling and model training
- CNN (Convolutional Neural Network): For image classification
- TensorFlow Serving: For model deployment
- FastAPI: For creating API endpoints
- React.js: For the web frontend
- Data Augmentation: To enhance the training dataset

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/potato-disease-prediction.git
   cd potato-disease-prediction
   ```

2. Set up the backend:
   - Install TensorFlow Serving
   - Start TensorFlow Serving with the trained model:
     ```
     tensorflow_model_server --model_base_path=/path/to/model --rest_api_port=8501
     ```
   - Install FastAPI and dependencies:
     ```
     pip install fastapi uvicorn
     ```
   - Start the FastAPI server:
     ```
     uvicorn main:app --reload
     ```

3. Set up the frontend:
   - Navigate to the React app directory:
     ```
     cd frontend
     ```
   - Install dependencies:
     ```
     npm install
     ```
   - Start the React development server:
     ```
     npm start
     ```

4. Access the web application at `http://localhost:3000`

## Usage

1. Open the web application in your browser.
2. Use the interface to capture or upload an image of a potato plant leaf.
3. The application will process the image and display whether the leaf is healthy or diseased.

## Future Improvements

- Expand the model to detect more types of plant diseases
- Improve model accuracy with more diverse training data
- Enhance the user interface for better user experience
- Implement user authentication and result history

