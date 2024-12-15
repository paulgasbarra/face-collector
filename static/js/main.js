document.addEventListener('DOMContentLoaded', function() {
    // Only initialize elements if we're on the upload page
    const uploadForm = document.getElementById('uploadForm');
    if (!uploadForm) return; // Exit if we're not on the upload page
    
    const imageInput = document.getElementById('imageInput');
    const uploadButton = document.getElementById('uploadButton');
    const spinner = uploadButton.querySelector('.spinner-border');
    const imageContainer = document.querySelector('.image-container');
    const originalImage = document.getElementById('originalImage');
    const processedImage = document.getElementById('processedImage');
    const errorAlert = document.getElementById('errorAlert');

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = imageInput.files[0];
        if (!file) {
            showError('Please select an image file.');
            return;
        }

        // Show loading state
        spinner.classList.remove('d-none');
        uploadButton.disabled = true;
        errorAlert.classList.add('d-none');

        const algorithm = document.getElementById('algorithm').value;
        const formData = new FormData();
        formData.append('file', file);
        formData.append('algorithm', algorithm);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }

            // Display images
            originalImage.src = data.original;
            processedImage.src = data.processed;
            imageContainer.classList.remove('d-none');
            
            // Display number of faces detected and matches
            const facesAlert = document.createElement('div');
            facesAlert.className = 'alert alert-info mt-3';
            
            let alertText = `${data.faces_detected} face${data.faces_detected !== 1 ? 's' : ''} detected and saved`;
            
            // Add matches information if available
            if (data.matches && Object.keys(data.matches).length > 0) {
                alertText += '<div class="mt-3"><h6>Potential Matches Found:</h6>';
                for (const [faceId, matches] of Object.entries(data.matches)) {
                    if (matches.length > 0) {
                        alertText += '<div class="matches-container mt-2">';
                        matches.forEach(match => {
                            const similarity = Math.round(match.similarity * 100);
                            const matchClass = similarity > 80 ? 'bg-success' : 
                                             similarity > 65 ? 'bg-info' : 'bg-warning';
                            alertText += `
                                <div class="match-item mb-2 p-2 border rounded">
                                    <div class="d-flex align-items-center">
                                        <span class="badge ${matchClass} me-2">
                                            ${similarity}% Match
                                        </span>
                                        <span class="me-2">with ${match.person_name}</span>
                                        <div class="btn-group ms-auto">
                                            <a href="/persons/${match.person_id}" 
                                               class="btn btn-sm btn-outline-primary">
                                                <i class="fas fa-user"></i> View Person
                                            </a>
                                            ${similarity > 80 ? `
                                            <a href="/persons/${match.person_id}/faces/${faceId}/assign" 
                                               class="btn btn-sm btn-outline-success"
                                               onclick="return confirm('Assign this face to ${match.person_name}?')">
                                                <i class="fas fa-link"></i> Quick Assign
                                            </a>` : ''}
                                        </div>
                                    </div>
                                    ${match.face_filename ? `
                                    <div class="mt-2">
                                        <small class="text-muted">
                                            Matched with photo from: ${match.face_filename}
                                        </small>
                                    </div>` : ''}
                                </div>`;
                        });
                        alertText += '</div>';
                    }
                }
                alertText += '</div>';
            }
            
            facesAlert.innerHTML = alertText;
            imageContainer.insertBefore(facesAlert, imageContainer.firstChild);

        } catch (error) {
            showError(error.message);
        } finally {
            // Reset loading state
            spinner.classList.add('d-none');
            uploadButton.disabled = false;
        }
    });

    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.classList.remove('d-none');
    }
});
