# Machine-Learning_OpenCV4
Machine Learning Exercises and Experiments with OpenCV 4

This repository contains exercises, experiments, and personal implementations based on the book "Machine Learning for OpenCV 4". It is intended as a practice workspace to understand, experiment, and fix problems in the concepts presented in the book, as well as to extend them with custom implementations.

📝 Overview
Implemented machine learning algorithms using OpenCV 4's ML module.
Focus on practical examples including:
SVM for pedestrian detection
HOG feature extraction
Sliding window object detection
Custom dataset handling (positive/negative samples)
Solving common issues encountered when following the book exercises.
Step-by-step implementation of exercises with detailed comments.

Dependencies
The code is primarily written in Python 3.11 with the following packages:
opencv-python (cv2)
numpy
matplotlib
scikit-learn (for accuracy evaluation and comparison)
pandas (for optional data handling)

🚀 Usage
Prepare positive and negative sample images in the data/ folder.
Adjust the win_size, block_size, and other HOG parameters in the scripts if needed.
Train a custom SVM model using train_svm.py (or similar scripts).
Test detection using sliding window or HOG’s detectMultiScale.
Results and annotated images will be saved in the results/ folder.

Features
Custom training of SVM for pedestrian detection.
Comparison between Sliding Window and OpenCV HOG detectMultiScale approaches.
Iterative improvement of the model by including false positives.
Fully commented code explaining each step of the workflow.
Ready for extension with other object detection or ML tasks.

References
Machine Learning for OpenCV 4 (Book)
OpenCV Official Documentation
scikit-learn Documentation

⚠️ Notes
Make sure all images are resized correctly to match the HOG win_size.
Some scripts rely on OpenCV's ML module (cv2.ml.SVM) rather than scikit-learn’s SVM for compatibility with HOG detectors.
The repository is mainly educational and for self-learning purposes.
