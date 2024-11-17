# Waste Classification System
## Abstract
The Waste Classification System addresses a critical environmental issue—efficient waste management. This system automates waste classification into four categories: Plastic, Glass, Paper, and Other. The system categorizes waste by utilizing deep learning, computer vision, and IoT technologies, promoting efficient recycling practices. The project integrates a ResNet50 deep learning model with a Raspberry Pi 4 and ultrasonic sensors for real-time waste detection and classification. The solution also features a smart cleaning schedule system that monitors bin fill levels and triggers notifications for timely collection, aligning with the sustainability goals of Saudi Vision 2030.

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Definition](#problem-definition)
3. [Aim of the Project](#aim-of-the-project)
4. [Objectives](#objectives)
5. [Model Evaluation](#model-evaluation)
6. [Project Structure](#project-structure)
7. [Technologies Used](#Technologies-Used)
8. [Acknowledgments](#Acknowledgments)

## Introduction

In today's world, efficient waste management is more critical than ever. Waste classification is essential for proper disposal, recycling, and resource recovery. This system aims to automate and optimize the process, reducing human intervention and ensuring that waste is sorted properly to enhance recycling efforts.

## Problem Definition

Urbanization and population growth lead to increasing waste generation, creating environmental and public health risks. In Saudi Arabia, manual waste sorting remains a labor-intensive and time-consuming task, which the **Ministry of Environment, Water, and Agriculture** seeks to address in alignment with **Saudi Vision 2030**.

## Aim of the Project

The primary goal of this project is to automate waste classification to help the government control waste management practices, promote a cleaner environment, and contribute to sustainable development. This project integrates real-time object detection, IoT capabilities, and energy-efficient systems to enhance the waste classification process.

## Objectives

1. **Promote environmental sustainability** by integrating **ultrasonic sensors** with **Raspberry Pi** and utilizing **IoT capabilities** in smart bins.
2. **Implement real-time waste classification** by accurately identifying and categorizing waste upon deposit.
3. Ensure **compliance with the Ministry of Environment, Water, and Agriculture** while supporting the **Saudi Vision 2030** goals for sustainable waste management.
4. **Design a low-power system** with **standby mode** activated by ultrasonic sensors for efficient energy use.
5. Conduct **extensive testing** to evaluate the system's performance and accuracy in real-world applications.

## Model Evaluation

This project explores the performance of multiple deep learning models, including:
- **Convolutional Neural Networks (CNN)**
- **ResNet50**
- **YOLOv8**

After evaluating the models based on **accuracy**, **speed**, and **efficiency**, **ResNet50** was selected for its superior performance in real-time classification. It was integrated with **Raspberry Pi 4** and paired with **ultrasonic sensors** for autonomous waste classification.

## Project Structure

The project consists of the following key components:
- **Deep Learning Model**: Trained on a diverse dataset of waste images for classification (Plastic, Glass, Paper, and Other).
- **Raspberry Pi 4**: Controls the waste classification system and operates in energy-efficient modes.
- **Ultrasonic Sensors**: Monitor bin fill levels and trigger cleaning schedule notifications.
- **IoT Integration**: Facilitates communication between smart bins and the backend system.

## Technologies Used
- Python 3.8+
- PyTorch
- TensorFlow
- OpenCV

## Acknowledgments
We would like to express our gratitude to God for His guidance and blessings throughout
this endeavor. We are also deeply grateful to our families for their constant love and
encouragement. Last but not least, we extend our sincere appreciation to our supervisor
Dr. Enas Jambi for her support and mentorship which elevated the quality of our paper.
