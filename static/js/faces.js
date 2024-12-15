document.addEventListener('DOMContentLoaded', function() {
    let currentFlippedCard = null;
    
    // Handle create person form submission
    const createPersonForm = document.getElementById('createPersonForm');
    if (createPersonForm) {
        createPersonForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            try {
                const response = await fetch('/persons/add', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok && data.redirect) {
                    window.location.href = data.redirect;
                } else {
                    throw new Error(data.error || 'Failed to create person');
                }
            } catch (error) {
                console.error('Error creating person:', error);
                alert(error.message);
            }
        });
    }
    
    // Set face ID when create person button is clicked
    document.querySelectorAll('.create-person').forEach(button => {
        button.addEventListener('click', function() {
            const faceId = this.dataset.faceId;
            document.getElementById('faceId').value = faceId;
        });
    });

    // Handle initial delete button click (flip animation)
    document.querySelectorAll('.delete-face').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            // Reset any previously flipped card
            if (currentFlippedCard && currentFlippedCard !== this.closest('.flip-card')) {
                currentFlippedCard.classList.remove('flipped');
            }
            
            const flipCard = this.closest('.flip-card');
            flipCard.classList.add('flipped');
            currentFlippedCard = flipCard;
        });
    });

    // Handle confirm delete button click
    document.querySelectorAll('.confirm-delete').forEach(button => {
        button.addEventListener('click', async function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            const faceId = this.dataset.faceId;
            const cardContainer = this.closest('.col-md-3');

            try {
                const response = await fetch(`/delete_face/${faceId}`, {
                    method: 'POST'
                });

                const data = await response.json();
                
                if (response.ok) {
                    // Animate the card removal
                    cardContainer.style.transition = 'opacity 0.3s ease-out';
                    cardContainer.style.opacity = '0';
                    setTimeout(() => cardContainer.remove(), 300);
                    currentFlippedCard = null;
                } else {
                    throw new Error(data.error || 'Failed to delete face');
                }
            } catch (error) {
                const flipCard = this.closest('.flip-card');
                flipCard.classList.remove('flipped');
                currentFlippedCard = null;
                console.error('Error deleting face:', error);
            }
        });
    });

    // Add document-level click handler to reset flipped cards
    document.addEventListener('click', function(e) {
        if (currentFlippedCard && !currentFlippedCard.contains(e.target)) {
            currentFlippedCard.classList.remove('flipped');
            currentFlippedCard = null;
        }
    });

    // Prevent clicks within flip-cards from triggering the document click handler
    document.querySelectorAll('.flip-card').forEach(card => {
        card.addEventListener('click', function(e) {
            e.stopPropagation();
        });
    });
});
