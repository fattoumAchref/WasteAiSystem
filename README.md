# DispoEasy

## Overview
WasteAI Assistant is an intelligent waste management solution developed as part of the coursework for the Advanced AI Applications module at ESPRIT Graduate School of Engineering in collaboration with Second Life. The project combines multiple AI technologies to help users identify, classify, and properly dispose of various waste materials.

## Features
- **Waste Classification**: Automatically identifies waste materials from uploaded images
- **Recycling Guidance**: Provides detailed information on how to properly recycle or dispose of identified waste
- **Waste Detection in Drone Footage**: Detects and localizes waste objects in environmental footage
- **AI-Powered Chatbot**: RAG-based assistant that answers questions about waste management and recycling
- **Sustainable Recycling Steps**: Creates  representations of recycled materials using stable diffusion .

## Project Documentation
- [Full Project Report](./report.pdf) - Comprehensive project documentation and analysis
- [Research Paper](./research_paper.pdf) - Academic paper detailing the AI approaches used
- [Project Blog](https://679fcb9642158.site123.me) - Follow our journey and updates on the Aspire platform

## Tech Stack

### Frontend
- **Framework**: Angular 14
- **UI Components**: Custom Angular components
- **Styling**: CSS with responsive design principles
- **HTTP Client**: Angular's HttpClient for API communication

### Backend
- **API Gateway**: FastAPI for routing and service orchestration
- **Service Discovery**: Consul for microservices management
- **Waste Classification**: EfficientNet model trained on waste dataset
- **Object Detection**: YOLOv8 for waste detection in environmental footage
- **Knowledge Base**: RAG (Retrieval Augmented Generation) system for waste information
- **Image Generation**: Stable Diffusion for creating recycling-themed visuals

## Directory Structure
```
WasteAiSystem/
├── apigateway/         # API Gateway service
├── wasteClassification/ # Waste classification model and service
├── RAGWaste/           # RAG-based chatbot for waste information
├── detection/          # YOLOv8 waste detection service
├── droneWasteClassif/  # Drone footage analysis
├── stablediffusion/    # Image generation service

waste_assistantfront/
├── src/                # Angular frontend source
    ├── app/            # Application components
        ├── components/ # UI components
            ├── home/           # Landing page
            ├── chatbot/        # RAG chatbot interface
            ├── detection/      # Waste detection UI
            ├── dronefootage/   # Drone footage analysis UI
```

## Getting Started

### Prerequisites
- Node.js 14+ and npm
- Python 3.8+
- CUDA-compatible GPU (recommended for detection and image generation)

### API Keys Setup
The project requires several API keys for external services. Before running the application, you need to create `.env` files in the following locations:

1. **Stable Diffusion Service**
   ```bash
   # In WasteAiSystem/stablediffusion/.env
   OPENAI_API_KEY=your_mistral_api_key_here
   ```

2. **RAG Waste Service**
   ```bash
   # In WasteAiSystem/RAGWaste/.env
   GOOGLE_BOOKS_API_KEY=your_google_books_api_key_here
   ```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/firasayari10/waste-ai-assistant.git
   cd waste-ai-assistant
   ```

2. **Backend Setup**
   ```bash
   cd WasteAiSystem
   # Install dependencies for all services
   cd apigateway && pip install -r requirements.txt
   cd ../wasteClassification && pip install -r requirements.txt
   cd ../RAGWaste && pip install -r requirements.txt
   cd ../detection && pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd waste_assistantfront
   npm install
   ```

### Running the Application

1. **Start Backend Services**
   ```bash
   # Start API Gateway
   cd WasteAiSystem/apigateway
   python apigateway.py
   
   # Start other services in separate terminals
   cd WasteAiSystem/wasteClassification
   python app.py
   
   cd WasteAiSystem/RAGWaste
   python app.py
   
   cd WasteAiSystem/detection
   python main.py
   ```

2. **Start Frontend**
   ```bash
   cd waste_assistantfront
   ng serve
   ```

3. **Access the Application**
   Open your browser and navigate to `http://localhost:4200`

## Acknowledgment
This project was developed as part of the Advanced AI Applications module at ESPRIT Graduate School of Engineering in collaboration with Second life (https://www.secondlife.ngo). We would like to thank our instructors for their guidance and support throughout the development process.