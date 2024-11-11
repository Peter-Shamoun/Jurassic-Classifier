document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const resetBtn = document.getElementById('reset-btn');
    const predictBtn = document.getElementById('predict-btn');
    const result = document.getElementById('result');
    const predictionText = document.getElementById('prediction-text');

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
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                dropZone.classList.add('hidden');
                previewContainer.classList.remove('hidden');
                predictBtn.disabled = false;
                result.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }
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
        predictBtn.disabled = true;
        predictBtn.textContent = 'Analyzing...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: previewImage.src
                })
            });

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            result.classList.remove('hidden');
            predictionText.textContent = data.prediction.replace(/_/g, ' ');
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = 'Identify Dinosaur';
        }
    }
}); 