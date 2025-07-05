# Robot_pet Project Documentation

## Project Overview

This project is an entry for the National College Students Embedded Competition Chip Application Track — **PawAI Companion: Interactive Smart Pet Companion System**.

This system is a complete intelligent pet interaction platform that enables natural interaction between humans and desktop electronic pets through face recognition, gesture recognition, cloud communication, and other technologies. The system supports multi-user management, features an affection cultivation mechanism, and provides cloud synchronization functionality, allowing users to monitor and control pet status in real-time through mobile apps.

## Core Values

- **Personalized Interaction**: Identifies different users through face recognition technology and maintains independent affection profiles for each user
- **Intelligent Response**: Enables natural human-computer interaction through gesture recognition, allowing pets to understand and respond to user actions
- **Cloud Synchronization**: Supports remote monitoring and control, users can stay informed about pet status anytime through Tuya Smart App
- **Cultivation Experience**: Provides continuous interaction motivation through the affection system, enhancing user engagement
- **High Extensibility**: Modular design supports feature expansion, such as voice interaction, additional sensor integration, etc.

---

## Main Features

### 🎯 User Identification and Management
- **Face Detection and Recognition**: Extracts facial feature vectors through face detection-alignment-recognition pipeline
- **Multi-user Support**: Automatically identifies different users and maintains independent affection data for each user
- **Data Persistence**: Stores user information and affection history based on SQLite database

### 🤖 Intelligent Interaction Control
- **Gesture Recognition**: Subscribes to `/hobot_hand_gesture_detection` topic for real-time user gesture recognition
- **Action Mapping**: Maps recognized gestures to specific robot control commands
- **Serial Communication**: Sends control commands to the lower computer via serial port to achieve robot action control
- **Real-time Feedback**: Receives responses from lower computer to ensure successful command execution

### 📊 Affection Cultivation System
- **UI Interface**: Displays pet affection, pet status, daily check-in and task completion
- **Automatic Growth**: Each successful interaction increases affection
- **Real-time Synchronization**: Affection changes are saved to local database in real-time
- **Cloud Commands**: Supports affection adjustment commands from cloud
- **Threshold Triggers**: Unlocks hidden features when affection reaches specific values

### ☁️ Cloud Communication and Monitoring
- **MQTT Protocol**: Implements reliable cloud communication based on Tuya cloud platform
- **Real-time Reporting**: Key status like affection is reported to cloud in real-time
- **Remote Control**: Supports remote monitoring and pet status adjustment through Tuya Smart App
- **Event-driven**: Responds immediately to cloud commands without polling

### 🎙️ Smart Conversation Feature
- **Conditional Unlock**: Automatically enables voice interaction when affection reaches threshold
- **Large Language Model**: Integrates large language models for intelligent conversation experience
- **Multimodal Interaction**: Combines multiple interaction methods like gestures and voice

### 🔧 System Architecture Features
- **Multi-threaded Design**: Serial monitoring, cloud communication run in independent threads to ensure system responsiveness
- **Event-driven**: Uses callback mechanisms to handle cloud commands, avoiding ineffective polling and improving efficiency
- **Modular Structure**: Each functional module is independently designed for easy maintenance and expansion
- **Fault Tolerance**: Comprehensive exception handling mechanisms ensure stable system operation

---

## Technology Stack

- **Programming Language**: Python 3
- **Robot Framework**: ROS 2 
- **Computer Vision**: OpenCV + Face Recognition Models
- **Database**: SQLite 3
- **Cloud Communication**: MQTT Protocol + Tuya IoT Platform
- **Hardware Interface**: Serial Communication
- **AI Models**: Integrated Large Language Model APIs


---

## File Structure

### Core Modules
- **`RDK/main/main.py`**  
  System main entry point, responsible for initializing ROS nodes and starting main functional modules

- **`RDK/main/PerceptionMonitor.py`**  
  Core control node, integrating face recognition, gesture recognition, serial communication, and cloud communication functions

### Functional Modules
- **`RDK/face_ws/opencv/face_sqlite.py`**  
  Local database operation interface, responsible for storing and managing user affection and facial feature vectors

- **`RDK/face_ws/opencv/inference.py`**  
  Face recognition inference module, implementing face detection, feature extraction, and identity recognition

- **`RDK/MQTT/tuya_mqtt.py`**  
  MQTT cloud communication module, handling data exchange and device control with Tuya IoT platform

- **`RDK/LLM_interface/voice_assistant_demo.py`**  
  Voice assistant module, integrating large language models for intelligent conversation functionality

- **`RDK/hand_ws/`**  
  Gesture recognition algorithm package, implementing gesture detection and recognition based on TROS algorithm framework

- **`RDK/UI/`**  
  UI interface #TODO Continuous improvement in progress

- **`STM32F103C8T6/test/Project`**  
  Keil project directory for STM32F103C8T6 microcontroller, containing firmware code for lower computer control
---

## Quick Start

### Environment Requirements (RDK_X5)
- Ubuntu 20.04 / 22.04
- ROS 2 Humble/Foxy
- Python 3.8+
- Supported camera device
- Serial device (/dev/ttyS1)

# Launch Project (RDK_X5)
cd /root/Robot_pet/main
python3 main.py

