# Face Detection and Recognition Web Application

A Flask-based web application for detecting and recognizing faces in images. The application supports multiple face detection algorithms and can match similar faces across different uploads.

## Features

- Multiple face detection algorithms:
  - Haar Cascade (Fast, good for frontal faces)
  - Deep Neural Network (More accurate, slower)
  - MTCNN (Best for varied angles and lighting)
- Face matching across uploads
- Person management system
- Interactive web interface
- Face similarity scoring

## Tech Stack

- Python 3.11
- Flask
- OpenCV
- TensorFlow
- MTCNN
- face_recognition
- PostgreSQL
- SQLAlchemy
- Bootstrap 5

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up the PostgreSQL database and update the DATABASE_URL environment variable.

4. Initialize the database:
```bash
flask db upgrade
```

5. Run the application:
```bash
python main.py
```

The application will be available at `http://localhost:5000`

## Project Structure

- `/static` - Static files (CSS, JS, uploaded images)
- `/templates` - HTML templates
- `/migrations` - Database migrations
- `app.py` - Main application file
- `models.py` - Database models
- `utils.py` - Utility functions for face detection and processing

## License

MIT License
