from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    faces = db.relationship('Face', backref='person', lazy=True)

    def __repr__(self):
        return f'<Person {self.name or "Unknown"}>'


class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(
        db.String(255),
        nullable=False)  # Format: face_YYYYMMDD_HHMMSS_index_originalname
    original_image = db.Column(db.String(255), nullable=False)
    captured_at = db.Column(db.DateTime, default=datetime.utcnow)
    person_id = db.Column(db.Integer,
                          db.ForeignKey('person.id'),
                          nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    embedding = db.Column(db.JSON,
                          nullable=True)  # Store face embedding as JSON

    def __repr__(self):
        return f'<Face {self.filename}>'

    def __init__(self,
                 filename=None,
                 original_image=None,
                 person_id=None,
                 confidence=None,
                 embedding=None):
        self.filename = filename
        self.original_image = original_image
        self.person_id = person_id
        self.confidence = confidence
        self.embedding = embedding
