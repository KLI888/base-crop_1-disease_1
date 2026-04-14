document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.getElementById('spinner');
    
    const resultContainer = document.getElementById('result-container');
    const recommendedCrop = document.getElementById('recommended-crop');
    const errorMessage = document.getElementById('error-message');
    const resultContent = document.querySelector('.result-content');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Loading State
        btnText.style.display = 'none';
        spinner.style.display = 'block';
        submitBtn.disabled = true;
        
        // Hide previous results
        resultContainer.classList.add('hidden');
        errorMessage.classList.add('hidden');
        resultContent.style.display = 'none';

        // Gather form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                // Success
                recommendedCrop.textContent = result.recommendation;
                resultContent.style.display = 'block';
            } else {
                // Error from server
                errorMessage.textContent = result.error || 'An unknown error occurred.';
                errorMessage.classList.remove('hidden');
            }
        } catch (error) {
            // Network or parsing error
            errorMessage.textContent = 'Failed to connect to the prediction service. Please ensure the server is running.';
            errorMessage.classList.remove('hidden');
        } finally {
            // Restore UI State
            resultContainer.classList.remove('hidden');
            btnText.style.display = 'block';
            spinner.style.display = 'none';
            submitBtn.disabled = false;
            
            // Re-trigger animation
            resultContainer.style.animation = 'none';
            resultContainer.offsetHeight; // trigger reflow
            resultContainer.style.animation = null;
        }
    });

    // Add lovely focus effects for inputs
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.parentElement.style.transform = 'translateY(-2px)';
            input.parentElement.style.transition = 'transform 0.3s ease';
        });
        
        input.addEventListener('blur', () => {
            input.parentElement.style.transform = 'translateY(0)';
        });
    });
});
