import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
import logging
from models import db, Person, Face
from utils import process_image, allowed_file, create_required_folders, find_similar_faces

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FACES_FOLDER'] = 'static/faces'

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Create the migration repository if it doesn't exist
with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        logging.error(f"Error creating database tables: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/faces')
def faces():
    faces = Face.query.order_by(Face.captured_at.desc()).all()
    logging.debug(f"Processing {len(faces)} faces for similarity matching")
    
    # Process similar faces for display
    for face in faces:
        if face.embedding:
            logging.debug(f"Processing face {face.id} with embedding")
            # Get all faces except the current one that are assigned to people
            other_faces = Face.query.filter(
                Face.id != face.id,
                Face.person_id.isnot(None)  # Changed from is_not to isnot
            ).all()
            logging.debug(f"Found {len(other_faces)} other faces to compare against")
            
            try:
                matches = find_similar_faces(face.embedding, other_faces)
                logging.debug(f"Found {len(matches)} matches for face {face.id}")
                
                if matches:
                    face.similar_faces = [
                        {
                            'person_id': match[0].person_id,
                            'person_name': match[0].person.name,
                            'similarity': match[1]
                        }
                        for match in matches[:3]  # Get top 3 matches
                    ]
                    logging.debug(f"Processed similar faces for face {face.id}: {face.similar_faces}")
                else:
                    face.similar_faces = []
            except Exception as e:
                logging.error(f"Error processing similar faces for face {face.id}: {str(e)}")
                face.similar_faces = []
        else:
            logging.debug(f"Face {face.id} has no embedding")
            face.similar_faces = []
            
    return render_template('faces.html', faces=faces)

@app.route('/persons')
def persons():
    persons = Person.query.order_by(Person.created_at.desc()).all()
    return render_template('persons.html', persons=persons)

@app.route('/persons/<int:person_id>')
def view_person(person_id):
    person = Person.query.get_or_404(person_id)
    unassigned_faces = Face.query.filter_by(person_id=None).all()
    return render_template('person_faces.html', person=person, unassigned_faces=unassigned_faces)

@app.route('/persons/add', methods=['POST'])
def add_person():
    try:
        name = request.form.get('name')
        face_id = request.form.get('face_id')
        
        if not name:
            if request.is_json:
                return jsonify({'error': 'Name is required'}), 400
            return redirect(url_for('persons'))

        try:
            person = Person(name=name)
            db.session.add(person)
            db.session.commit()  # Commit first to get the person.id

            if face_id:
                face = Face.query.get(face_id)
                if face:
                    face.person_id = person.id
                    db.session.commit()

            if request.is_json:
                return jsonify({'success': True, 'redirect': url_for('persons')})
            return redirect(url_for('persons'))
            
        except Exception as e:
            logging.error(f"Error adding person: {str(e)}")
            db.session.rollback()
            if request.is_json:
                return jsonify({'error': str(e)}), 500
            return redirect(url_for('persons'))
    except Exception as e:
        logging.error(f"Error adding person: {str(e)}")
        db.session.rollback()
        if request.is_json:
            return jsonify({'error': str(e)}), 500
        return redirect(url_for('persons'))

@app.route('/persons/<int:person_id>/faces/<int:face_id>/assign', methods=['POST'])
def assign_face(person_id, face_id):
    try:
        person = Person.query.get_or_404(person_id)
        face = Face.query.get_or_404(face_id)
        face.person_id = person.id
        db.session.commit()
        return redirect(url_for('view_person', person_id=person_id))
    except Exception as e:
        logging.error(f"Error assigning face: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Error assigning face'}), 500

@app.route('/persons/<int:person_id>/faces/<int:face_id>/unassign', methods=['POST'])
def unassign_face(person_id, face_id):
    try:
        person = Person.query.get_or_404(person_id)
        face = Face.query.get_or_404(face_id)
        if face.person_id == person.id:
            face.person_id = None
            db.session.commit()
        return redirect(url_for('view_person', person_id=person_id))
    except Exception as e:
        logging.error(f"Error unassigning face: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Error unassigning face'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.debug("Upload request received")

    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Create folders if they don't exist
            create_required_folders()

            if file.filename is not None:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                logging.debug(f"Saving file to: {filepath}")
                file.save(filepath)
            else:
                logging.error("No filename provided")
                return jsonify({'error': 'No filename provided'}), 400

            # Get algorithm choice
            algorithm = request.form.get('algorithm', 'haar')
            logging.debug(f"Selected algorithm: {algorithm}")

            # Process image for face detection and save faces
            processed_filename, saved_faces = process_image(
                filepath, algorithm, app.config['FACES_FOLDER'])
            logging.debug(f"Processed image saved as: {processed_filename}")
            logging.debug(f"Detected {len(saved_faces)} faces")

            # Save face entries to database and find matches
            saved_face_ids = []
            matches_by_face = {}
            
            for face_filename, face_embedding in saved_faces:
                try:
                    # First, look for similar faces among existing faces
                    if face_embedding:
                        existing_faces = Face.query.filter(Face.person_id.isnot(None)).all()
                        matches = find_similar_faces(face_embedding, existing_faces, threshold=0.6)
                        
                        if matches:
                            logging.debug(f"Found {len(matches)} potential matches for {face_filename}")
                            matches_by_face[face_filename] = matches
                    
                    # Save the new face
                    face = Face(
                        filename=face_filename,
                        original_image=filename,
                        embedding=face_embedding
                    )
                    db.session.add(face)
                    db.session.commit()
                    saved_face_ids.append(face.id)
                    logging.debug(f"Successfully saved face: {face_filename}")
                except Exception as e:
                    logging.error(f"Error saving face {face_filename}: {str(e)}")
                    db.session.rollback()
                    continue
            
            if saved_face_ids:
                logging.debug(f"Successfully saved {len(saved_face_ids)} faces to database")
            else:
                logging.warning("No faces were saved to database")
            
            # Find similar faces for each newly saved face
            similar_faces = {}
            try:
                for face_id in saved_face_ids:
                    face = Face.query.get(face_id)
                    if face and face.embedding:
                        # Get all faces except the current one that are assigned to people
                        other_faces = Face.query.filter(
                            Face.id != face.id,
                            Face.person_id.is_not(None)  # Using is_not instead of isnot
                        ).all()
                        
                        matches = find_similar_faces(face.embedding, other_faces)
                        if matches:
                            similar_faces[face.id] = [
                                {
                                    'id': match[0].id,
                                    'person_id': match[0].person_id,
                                    'person_name': match[0].person.name if match[0].person else None,
                                    'similarity': match[1]
                                }
                                for match in matches[:3]  # Get top 3 matches
                            ]
                            logging.debug(f"Found {len(matches)} similar faces for face {face_id}")
            except Exception as e:
                logging.error(f"Error finding similar faces: {str(e)}")

            # Prepare matches information for response
            matches_info = {}
            for face_id in saved_face_ids:
                face = Face.query.get(face_id)
                if face and face.filename in matches_by_face:
                    matches = matches_by_face[face.filename]
                    matches_info[face_id] = [
                        {
                            'id': match[0].id,
                            'person_id': match[0].person_id,
                            'person_name': match[0].person.name,
                            'similarity': float(match[1]),
                            'face_filename': match[0].filename
                        }
                        for match in matches
                    ]

            return jsonify({
                'original': f'/static/uploads/{filename}',
                'processed': f'/static/uploads/{processed_filename}',
                'faces_detected': len(saved_faces),
                'matches': matches_info
            })
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}", exc_info=True)
            db.session.rollback()
            return jsonify({'error': str(e)}), 500

    logging.error(f"Invalid file type: {file.filename}")
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/delete_face/<int:face_id>', methods=['POST'])
def delete_face(face_id):
    try:
        face = Face.query.get_or_404(face_id)

        # Delete the actual file
        face_path = os.path.join(app.config['FACES_FOLDER'], face.filename)
        if os.path.exists(face_path):
            os.remove(face_path)

        # Remove from database
        db.session.delete(face)
        db.session.commit()

        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error deleting face: {str(e)}")
        return jsonify({'error': 'Error deleting face'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)