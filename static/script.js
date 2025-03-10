document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const resetBtn = document.getElementById('reset-btn');
    const predictBtn = document.getElementById('predict-btn');
    const result = document.getElementById('result');
    const predictionText = document.getElementById('prediction-text');
    
    // Constants
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    
    // Add confidence display elements
    const resultContent = document.querySelector('.result-content');
    if (!document.getElementById('confidence-container')) {
        const confidenceContainer = document.createElement('div');
        confidenceContainer.id = 'confidence-container';
        confidenceContainer.className = 'confidence-container';
        confidenceContainer.innerHTML = `
            <div class="confidence-meter">
                <div id="confidence-bar" class="confidence-bar"></div>
            </div>
            <p id="confidence-text" class="confidence-text"></p>
            <div id="top-predictions" class="top-predictions"></div>
        `;
        resultContent.appendChild(confidenceContainer);
    }
    
    // Add version info element at the bottom of the card
    const card = document.querySelector('.card');
    if (!document.getElementById('version-info')) {
        const versionInfo = document.createElement('div');
        versionInfo.id = 'version-info';
        versionInfo.className = 'version-info';
        card.appendChild(versionInfo);
        
        // Fetch model info when page loads
        fetchModelInfo();
    }

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file);
    });

    resetBtn.addEventListener('click', resetUpload);
    predictBtn.addEventListener('click', predict);

    function handleFile(file) {
        // Validate file
        if (!file) return;
        
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file (JPEG, PNG, etc.)');
            return;
        }
        
        // Check file size
        if (file.size > MAX_FILE_SIZE) {
            showError(`File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB.`);
            return;
        }
        
        // Show loading state
        previewImage.src = '';
        previewImage.alt = 'Loading preview...';
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        previewContainer.classList.add('loading');
        
        // Create low-quality preview immediately
        const reader = new FileReader();
        reader.onload = function(e) {
            // Show low-res preview first
            const img = new Image();
            img.onload = function() {
                // Create a canvas for low-quality preview
                const canvas = document.createElement('canvas');
                canvas.width = 100;
                canvas.height = 100 * (img.height / img.width);
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                // Show low-res preview
                previewImage.src = canvas.toDataURL();
                previewImage.alt = 'Image preview';
                
                // Then load full image
                const fullReader = new FileReader();
                fullReader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewContainer.classList.remove('loading');
                    predictBtn.disabled = false;
                    result.classList.add('hidden');
                };
                fullReader.readAsDataURL(file);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        fileInput.value = '';
        previewImage.src = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        predictBtn.disabled = true;
        result.classList.add('hidden');
    }

    async function predict() {
        // Show loading state
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="btn-text">Analyzing...</span><div class="spinner"></div>';
        result.classList.add('hidden');
        
        try {
            // Get CSRF token if present
            const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content;
            
            const headers = {
                'Content-Type': 'application/json',
            };
            
            // Add CSRF token if available
            if (csrfToken) {
                headers['X-CSRFToken'] = csrfToken;
            }
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({
                    image: previewImage.src
                })
            });

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Display results
            showResults(data);
        } catch (error) {
            showError(error.message);
        } finally {
            // Reset button
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<span class="btn-text">Identify Dinosaur</span><span class="btn-icon">🔍</span>';
        }
    }
    
    function showResults(data) {
        const confidenceBar = document.getElementById('confidence-bar');
        const confidenceText = document.getElementById('confidence-text');
        const topPredictions = document.getElementById('top-predictions');
        
        // Set main prediction
        predictionText.textContent = data.prediction;
        
        // Set confidence
        const confidence = Math.round(data.confidence * 100);
        confidenceBar.style.width = `${confidence}%`;
        confidenceText.textContent = `Confidence: ${confidence}%`;
        
        // Get confidence class
        let confidenceClass = 'low';
        if (confidence >= 90) {
            confidenceClass = 'high';
        } else if (confidence >= 70) {
            confidenceClass = 'medium';
        }
        confidenceBar.className = `confidence-bar ${confidenceClass}`;
        
        // Show top predictions if available
        if (data.top_predictions && data.top_predictions.length > 1) {
            let topPredictionsHtml = '<h3>Alternative matches:</h3><ul>';
            
            // Skip the first prediction (already shown as the main result)
            for (let i = 1; i < data.top_predictions.length; i++) {
                const pred = data.top_predictions[i];
                const altConfidence = Math.round(pred.confidence * 100);
                topPredictionsHtml += `<li>${pred.species} (${altConfidence}%)</li>`;
            }
            
            topPredictionsHtml += '</ul>';
            topPredictions.innerHTML = topPredictionsHtml;
            topPredictions.style.display = 'block';
        } else {
            topPredictions.style.display = 'none';
        }
        
        // Display processing time if available
        if (data.processing_time) {
            const processingTime = document.createElement('p');
            processingTime.className = 'processing-time';
            processingTime.textContent = `Processing time: ${data.processing_time.toFixed(2)}s`;
            topPredictions.appendChild(processingTime);
        }
        
        // Show the result container
        result.classList.remove('hidden');
    }
    
    function showError(message) {
        result.classList.remove('hidden');
        predictionText.innerHTML = `<span class="error">Error: ${message}</span>`;
        document.getElementById('confidence-container').style.display = 'none';
    }
    
    async function fetchModelInfo() {
        try {
            const response = await fetch('/api/model-info');
            const data = await response.json();
            
            if (data.error) {
                console.error('Error fetching model info:', data.error);
                return;
            }
            
            // Display version info
            const versionInfo = document.getElementById('version-info');
            versionInfo.innerHTML = `
                <p>Model Version: ${data.version} | Classes: ${data.classes} | Last Loaded: ${data.last_loaded || 'N/A'}</p>
            `;
        } catch (error) {
            console.error('Error fetching model info:', error);
        }
    }
}); 