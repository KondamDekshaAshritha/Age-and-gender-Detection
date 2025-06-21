# Age-and-gender-Detection
Train a Convolutional Neural Network (CNN) on the *UTKFace* dataset to *predict multiple outputs*:

* 🧓 *Age* (regression)
* 🚻 *Gender* (binary classification)
* 🌍 *Ethnicity* (multi-class classification: 5 classes)
* 🔍 *Filtered/Unfiltered* label (binary classification)

---

## 🧩 *Code Breakdown*

### 1. *📦 Importing Modules*

* Data handling: pandas, numpy
* Visualization: matplotlib, seaborn
* Image processing: PIL, keras.preprocessing.image
* Model building: tensorflow.keras
* Others: tqdm for progress bar, warnings suppression

---

### 2. *📂 Loading the UTKFace Dataset*

* Reads images from the given directory.
* Extracts metadata (age, gender, ethnicity) from filenames.
* Adds a new column filtered based on presence of the word "filtered" in filename.
* Stores all data in a Pandas DataFrame.

---

### 3. *📊 Data Visualization*

* Displays:

  * A sample image with its labels.
  * Histograms and count plots for age, gender, ethnicity, and filtered data.
  * A grid of 25 sample images with their labels.

---

### 4. *🛠 Preprocessing*

* Loads all images in grayscale.
* Resizes to 128×128.
* Normalizes pixel values to \[0, 1].
* Reshapes to match CNN input.
* Encodes ethnicity as one-hot (5 classes).

---

### 5. *🧠 CNN Model Architecture*

* Shared CNN feature extractor (Conv → MaxPool layers).
* *Four output heads*:

  * gender_out: Sigmoid layer (binary classification)
  * age_out: ReLU layer (regression)
  * ethnicity_out: Softmax layer (multi-class)
  * filter_out: Sigmoid layer (binary classification)

---

### 6. *⚙ Model Compilation & Training*

* Loss functions:

  * Gender & Filtered: binary_crossentropy
  * Age: mae (Mean Absolute Error)
  * Ethnicity: categorical_crossentropy
* Optimizer: adam
* Metrics: accuracy for classification, mae for age
* Trained for 10 epochs with validation_split=0.2

---

### 7. *🧪 Image Prediction Extension*

* Allows image upload (e.g., in Colab).
* Preprocesses uploaded image similarly to training data.
* Predicts all 4 attributes using the trained model.
* Displays predicted labels and the input image.

---

