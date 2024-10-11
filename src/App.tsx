import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

// Define the shape of the history items
interface HistoryItem {
  imageURL: string;
  predictedClass: string;
}

function App() {
  const [model, setModel] = useState<tf.LayersModel | null>(null); // State to store the loaded model
  const [imageURL, setImageURL] = useState<string | null>(null); // State to store uploaded image URL
  const [prediction, setPrediction] = useState<string>(""); // State to store the predicted result
  const [history, setHistory] = useState<HistoryItem[]>([]); // State to store history of predictions

  // Update class labels if necessary
  const classLabels: string[] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
  ];

  // Load the TensorFlow.js model when the app loads
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log("Attempting to load model from /model/model.json");
        const loadedModel = await tf.loadLayersModel(`${import.meta.env.BASE_URL}model/model.json`, { strict: false });
        setModel(loadedModel);
        console.log("Model loaded successfully:", loadedModel);
      } catch (error) {
        console.error("Error loading model:", error);
      }
    };

    loadModel();
  }, []);

  // Handle file upload and display image
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImageURL(reader.result as string); // Store the image URL for preview
      };
      reader.readAsDataURL(file);
    }
  };

  // Make a prediction when the image is uploaded
  const handlePredict = async () => {
    if (!model || !imageURL) {
      alert("Please upload an image and wait for the model to load.");
      return;
    }

    const img = new Image();
    img.src = imageURL;
    img.onload = async () => {
      const tensorImg = preprocessImage(img); // Preprocess image to match model input
      const prediction = await model.predict(tensorImg) as tf.Tensor; // Make a prediction
      const predictionData = await prediction.data();
      const predictedClass = classLabels[argMax(Array.from(predictionData))];
      setPrediction(predictedClass); // Update state with predicted class

      // Add the current prediction and image to the history
      setHistory([...history, { imageURL, predictedClass }]);
    };
  };

  // Preprocess the image to fit the model input size (224x224 or any new size)
  // Preprocess the image to fit the model input size (28x28 and grayscale)
const preprocessImage = (img: HTMLImageElement) => {
  const tensor = tf.browser.fromPixels(img, 1) // 1 channel for grayscale
    .resizeNearestNeighbor([28, 28]) // Resize to 28x28 pixels
    .toFloat()
    .div(tf.scalar(255.0)) // Normalize the image
    .expandDims(); // Add a batch dimension (1, 28, 28, 1)
  return tensor;
};


  // Utility function to find the index of the max value
  const argMax = (arr: number[]) => {
    return arr.indexOf(Math.max(...arr));
  };

  return (
    <div className="App">
      <div className="flex items-center p-2 border-b border-black">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="30px" height="20px">
          <path d="M184 0c30.9 0 56 25.1 56 56l0 400c0 30.9-25.1 56-56 56c-28.9 0-52.7-21.9-55.7-50.1c-5.2 1.4-10.7 2.1-16.3 2.1c-35.3 0-64-28.7-64-64c0-7.4 1.3-14.6 3.6-21.2C21.4 367.4 0 338.2 0 304c0-31.9 18.7-59.5 45.8-72.3C37.1 220.8 32 207 32 192c0-30.7 21.6-56.3 50.4-62.6C80.8 123.9 80 118 80 112c0-29.9 20.6-55.1 48.3-62.1C131.3 21.9 155.1 0 184 0zM328 0c28.9 0 52.6 21.9 55.7 49.9c27.8 7 48.3 32.1 48.3 62.1c0 6-.8 11.9-2.4 17.4c28.8 6.2 50.4 31.9 50.4 62.6c0 15-5.1 28.8-13.8 39.7C493.3 244.5 512 272.1 512 304c0 34.2-21.4 63.4-51.6 74.8c2.3 6.6 3.6 13.8 3.6 21.2c0 35.3-28.7 64-64 64c-5.6 0-11.1-.7-16.3-2.1c-3 28.2-26.8 50.1-55.7 50.1c-30.9 0-56-25.1-56-56l0-400c0-30.9 25.1-56 56-56z"/>
        </svg>
        <h1 className="ml-2 text-lg font-bold">Clothing Classifier</h1>
      </div>

      <div className="px-[150px] pt-[40px] ">
        <h1 className="text-[30px] font-semibold">Clothing Classifier</h1>
        <p className="pt-2 text-[20px] font-light">
          This web application uses a Convolutional Neural Network (CNN) model to classify different types of clothing from images. The model can detect items such as:
          T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.
          <div className="flex py-4 gap-4">          
          </div>
        </p>

        {/* Upload Image Button */}
        <div className="px-[150px] py-[20px] flex items-center gap-4">
          {/* File Input */}
          <label className="flex flex-col min-w-40 flex-1 w-full resize-none overflow-hidden rounded-full text-[#1C160C] focus:outline-none focus:ring-0 border border-gray-600 bg-[#FFFFFF] placeholder:text-[#A18249] p-2 text-[14px]">
            <input
              type="file"
              accept="image/*"
              className="hidden"
              placeholder="Upload image"
              onChange={handleImageUpload}
            />
            <p className="px-4">Click to Upload Image</p>
          </label>

          {/* Upload and Predict Button */}
          <button
            onClick={handlePredict}
            className="bg-[#019863] text-white text-base font-bold py-2 px-6 rounded-full hover:bg-green-700 transition duration-300 ease-in-out"
          >
            Predict
          </button>
        </div>

        {/* Display Uploaded Image */}
        {imageURL && (
          <div className="flex justify-center mt-4">
            <img src={imageURL} alt="Uploaded" className="w-auto h-64 object-cover rounded-lg shadow-md" />
          </div>
        )}

        {/* Prediction Result */}
        {prediction && (
          <div className="mt-4">
            <h2 className="text-2xl font-bold text-gray-800 mb-3">Prediction: {prediction}</h2>
          </div>
        )}

        {/* History of Predictions */}
        {history.length > 0 && (
          <div className="mt-4">
            <h3 className="text-xl font-bold text-gray-700 mb-4">Prediction History</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {history.map((item, index) => (
                <div key={index} className="relative rounded-lg shadow-md overflow-hidden">
                  <img src={item.imageURL} alt={`Prediction: ${item.predictedClass}`} className="w-full h-[150px] object-cover" />
                  <div className="absolute bottom-0 left-0 right-0 p-2 text-white text-sm font-bold">
                    Previous Prediction: {item.predictedClass}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
