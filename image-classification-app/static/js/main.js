// Main JavaScript for Forest Fire Detection Web App

document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss flash messages
    setTimeout(function() {
        const flashMessages = document.querySelectorAll('.flash-message');
        flashMessages.forEach(function(msg) {
            msg.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => msg.remove(), 300);
        });
    }, 5000);
    
    // File input previews
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(function(input) {
        input.addEventListener('change', function(e) {
            const preview = this.parentElement.querySelector('.file-preview');
            if (!preview) return;
            
            const files = e.target.files;
            
            if (files.length === 0) {
                preview.innerHTML = '<i class="fas fa-image" style="font-size: 48px;"></i><p>No file selected</p>';
                return;
            }
            
            if (files.length === 1) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                    };
                    reader.readAsDataURL(file);
                } else {
                    preview.innerHTML = `<i class="fas fa-file"></i><p>${file.name}</p>`;
                }
            } else {
                preview.innerHTML = `<i class="fas fa-images"></i><p>${files.length} files selected</p>`;
            }
        });
    });
});

// Webcam functionality
let webcamStream = null;
let webcamInterval = null;

function startWebcam() {
    const video = document.getElementById('webcamVideo');
    
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            webcamStream = stream;
            video.srcObject = stream;
            video.play();
            
            // Start detection
            webcamInterval = setInterval(processWebcamFrame, 1000);
        })
        .catch(function(err) {
            console.error('Webcam error:', err);
            alert('Cannot access webcam: ' + err.message);
        });
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    if (webcamInterval) {
        clearInterval(webcamInterval);
        webcamInterval = null;
    }
}

function processWebcamFrame() {
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    const base64Data = imageData.split(',')[1];
    
    fetch('/api/webcam_frame', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Data })
    })
    .then(response => response.json())
    .then(data => {
        updateWebcamOverlay(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function updateWebcamOverlay(data) {
    const overlay = document.getElementById('webcamOverlay');
    if (!overlay) return;
    
    let className = '';
    let icon = '';
    
    if (data.predicted_class === 'fire') {
        className = 'prediction-fire';
        icon = '<i class="fas fa-fire"></i>';
    } else if (data.predicted_class === 'smoke') {
        className = 'prediction-smoke';
        icon = '<i class="fas fa-smog"></i>';
    } else {
        className = 'prediction-safe';
        icon = '<i class="fas fa-check-circle"></i>';
    }
    
    overlay.innerHTML = `
        <div class="${className}" style="padding: 10px; border-radius: 5px;">
            ${icon} ${data.predicted_class.toUpperCase()}: ${(data.confidence * 100).toFixed(1)}%
        </div>
    `;
}

// Chart creation helper
function createChart(canvasId, type, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
            }
        }
    };
    
    return new Chart(ctx, {
        type: type,
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#2ecc71',
                    '#e74c3c',
                    '#f39c12',
                    '#3498db',
                    '#9b59b6'
                ],
                borderWidth: 1
            }]
        },
        options: Object.assign(defaultOptions, options)
    });
}

// Add CSS animation for slideOut
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);