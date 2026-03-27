/**
 * NASA Exoplanet Classification System - Frontend JavaScript
 * Professional web application with API integration, Chart.js, and modern UI interactions
 * 
 * Author: NASA Space Apps Challenge Team
 * Version: 1.0.0
 */

// Global variables and configuration
const API_BASE_URL = '';
const SUPPORTED_DATASETS = ['k2', 'tess', 'koi'];
let currentDataset = 'k2';
let classDistributionChart = null;
let isLoading = false;
let highContrastMode = false;

// Accessibility and user preferences
const userPreferences = {
    highContrast: false,
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    fontSize: 'normal'
};

// DOM element references
const elements = {
    // Dataset selector
    datasetSelect: document.getElementById('dataset-select'),
    
    // Forms
    manualForm: document.getElementById('manual-form'),
    uploadForm: document.getElementById('upload-form'),
    retrainForm: document.getElementById('retrain-form'),
    
    // Input fields
    plOrbper: document.getElementById('pl_orbper'),
    plTrandep: document.getElementById('pl_trandep'),
    stTeff: document.getElementById('st_teff'),
    
    // File upload
    dropZone: document.getElementById('drop-zone'),
    csvFile: document.getElementById('csv-file'),
    browseBtn: document.getElementById('browse-btn'),
    retrainCheckbox: document.getElementById('retrain-checkbox'),
    
    // Buttons
    predictBtn: document.getElementById('predict-btn'),
    uploadBtn: document.getElementById('upload-btn'),
    retrainBtn: document.getElementById('retrain-btn'),
    
    // Results containers
    manualResults: document.getElementById('manual-results'),
    uploadResults: document.getElementById('upload-results'),
    retrainResults: document.getElementById('retrain-results'),
    noRetrain: document.getElementById('no-retrain'),
    
    // Loading overlay
    loadingOverlay: document.getElementById('loading-overlay'),
    
    // Toast container
    toastContainer: document.getElementById('toast-container'),
    
    // Statistics elements
    accuracyValue: document.getElementById('accuracy-value'),
    totalSamples: document.getElementById('total-samples'),
    modelType: document.getElementById('model-type'),
    
    // Sliders
    nEstimatorsSlider: document.getElementById('n_estimators'),
    maxDepthSlider: document.getElementById('max_depth'),
    learningRateSlider: document.getElementById('learning_rate'),
    
    // Slider value displays
    nEstimatorsValue: document.getElementById('n_estimators_value'),
    maxDepthValue: document.getElementById('max_depth_value'),
    learningRateValue: document.getElementById('learning_rate_value'),
    
    // Accessibility elements
    highContrastToggle: document.getElementById('high-contrast-toggle'),
    liveRegion: document.getElementById('live-region')
};

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Global error handler for unhandled promise rejections and async errors
window.addEventListener('unhandledrejection', function(event) {
    console.warn('Unhandled promise rejection:', event.reason);
    // Prevent the default browser behavior of logging to console
    event.preventDefault();
});

// Global error handler for general errors
window.addEventListener('error', function(event) {
    console.warn('Global error caught:', event.error);
    // Don't prevent default for general errors, just log them
});

/**
 * Initialize all application components
 */
function initializeApp() {
    initializeEventListeners();
    initializeSliders();
    initializeFileUpload();
    initializeAccessibility();
    
    // Initialize current dataset from the selector
    if (elements.datasetSelect) {
        currentDataset = elements.datasetSelect.value;
        console.log('Initial dataset set to:', currentDataset);
        // Show/hide TESS fields based on initial dataset
        handleDatasetChange();
    } else {
        console.error('Dataset selector not found!');
    }
    
    loadStatistics();
    updateDatasetSelector();
    
    // Check for saved user preferences
    loadUserPreferences();
    
    // Announce initialization to screen readers
    announceToScreenReader('NASA Exoplanet Classification System initialized successfully');
    
    console.log('NASA Exoplanet Classification System initialized successfully');
    showToast('System initialized successfully', 'success');
}

/**
 * Initialize all event listeners
 */
function initializeEventListeners() {
    // Dataset selector
    elements.datasetSelect.addEventListener('change', handleDatasetChange);
    
    // Form submissions
    elements.manualForm.addEventListener('submit', handleManualPrediction);
    elements.uploadForm.addEventListener('submit', handleFileUpload);
    elements.retrainForm.addEventListener('submit', handleModelRetrain);
    
    // Input validation
    [elements.plOrbper, elements.plTrandep, elements.stTeff].forEach(input => {
        input.addEventListener('blur', validateInput);
        input.addEventListener('input', clearValidation);
    });
    
    // File upload checkbox
    elements.retrainCheckbox.addEventListener('change', handleRetrainCheckboxChange);
    
    // Accessibility controls
    if (elements.highContrastToggle) {
        elements.highContrastToggle.addEventListener('click', toggleHighContrast);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', handleKeyboardNavigation);
}

/**
 * Initialize slider functionality
 */
function initializeSliders() {
    const sliders = [elements.nEstimatorsSlider, elements.maxDepthSlider, elements.learningRateSlider];
    const valueDisplays = [elements.nEstimatorsValue, elements.maxDepthValue, elements.learningRateValue];
    
    sliders.forEach((slider, index) => {
        slider.addEventListener('input', () => updateSliderValue(slider, valueDisplays[index]));
        // Set initial values
        updateSliderValue(slider, valueDisplays[index]);
    });
}

/**
 * Update slider value display
 */
function updateSliderValue(slider, valueDisplay) {
    const value = slider.value;
    if (slider.name === 'learning_rate') {
        valueDisplay.textContent = parseFloat(value).toFixed(2);
    } else {
        valueDisplay.textContent = value;
    }
}

/**
 * Initialize file upload functionality
 */
function initializeFileUpload() {
    // Browse button
    elements.browseBtn.addEventListener('click', () => {
        elements.csvFile.click();
    });
    
    // File input change
    elements.csvFile.addEventListener('change', handleFileSelect);
    
    // Drag and drop functionality
    elements.dropZone.addEventListener('dragover', handleDragOver);
    elements.dropZone.addEventListener('dragleave', handleDragLeave);
    elements.dropZone.addEventListener('drop', handleDrop);
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        elements.dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
}

/**
 * Prevent default drag behaviors
 */
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

/**
 * Handle drag over event
 */
function handleDragOver(e) {
    elements.dropZone.classList.add('dragover');
}

/**
 * Handle drag leave event
 */
function handleDragLeave(e) {
    elements.dropZone.classList.remove('dragover');
}

/**
 * Handle drop event
 */
function handleDrop(e) {
    elements.dropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

/**
 * Handle file selection
 */
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

/**
 * Handle file validation and setup
 */
function handleFile(file) {
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('Please select a CSV file', 'error');
        return;
    }
    
    // Validate file size (16MB limit)
    if (file.size > 16 * 1024 * 1024) {
        showToast('File size must be less than 16MB', 'error');
        return;
    }
    
    // Enable upload button
    elements.uploadBtn.disabled = false;
    elements.uploadBtn.classList.remove('disabled:opacity-50', 'disabled:cursor-not-allowed');
    
    // Update drop zone text
    elements.dropZone.querySelector('p').textContent = `Selected: ${file.name}`;
    
    showToast('File selected successfully', 'success');
}

/**
 * Handle dataset change
 */
function handleDatasetChange() {
    currentDataset = elements.datasetSelect.value;
    loadStatistics();
    
    // Show/hide TESS-specific fields
    const tessFields = document.getElementById('tess-fields');
    if (tessFields) {
        if (currentDataset === 'tess') {
            tessFields.style.display = 'grid';
            // Make TESS fields required
            document.getElementById('pl_trandurh').required = true;
            document.getElementById('st_pmralim').required = true;
            document.getElementById('st_pmdeclim').required = true;
        } else {
            tessFields.style.display = 'none';
            // Make TESS fields not required
            document.getElementById('pl_trandurh').required = false;
            document.getElementById('st_pmralim').required = false;
            document.getElementById('st_pmdeclim').required = false;
        }
    }
    
    // Update template download link
    const templateLink = document.getElementById('template-download-link');
    const templateText = document.getElementById('template-download-text');
    if (templateLink && templateText) {
        templateLink.href = `/download-template?dataset=${currentDataset}`;
        templateText.textContent = `Download ${currentDataset.toUpperCase()} Template`;
    }
    
    showToast(`Switched to ${currentDataset.toUpperCase()} dataset`, 'info');
}

/**
 * Update dataset selector options
 */
function updateDatasetSelector() {
    elements.datasetSelect.innerHTML = `
        <option value="k2">K2 Mission</option>
        <option value="tess">TESS Mission</option>
        <option value="koi">Kepler Objects of Interest</option>
    `;
    elements.datasetSelect.value = currentDataset;
}

/**
 * Handle manual prediction form submission
 */
async function handleManualPrediction(event) {
    event.preventDefault();
    
    if (isLoading) return;
    
    // Validate all inputs
    let inputs = [elements.plOrbper, elements.plTrandep, elements.stTeff];
    
    // Add TESS fields if dataset is TESS
    if (currentDataset === 'tess') {
        inputs.push(
            document.getElementById('pl_trandurh'),
            document.getElementById('st_pmralim'),
            document.getElementById('st_pmdeclim')
        );
    }
    
    let isValid = true;
    
    inputs.forEach(input => {
        if (!validateInput({ target: input })) {
            isValid = false;
        }
    });
    
    if (!isValid) {
        showToast('Please fix validation errors', 'error');
        return;
    }
    
    // Collect form data based on dataset
    const data = {
        dataset: currentDataset
    };
    
    // Add common features
    data.pl_orbper = parseFloat(elements.plOrbper.value);
    data.pl_trandep = parseFloat(elements.plTrandep.value);
    data.st_teff = parseFloat(elements.stTeff.value);
    
    // Add TESS-specific features if dataset is TESS
    if (currentDataset === 'tess') {
        const pl_trandurh = document.getElementById('pl_trandurh');
        const st_pmralim = document.getElementById('st_pmralim');
        const st_pmdeclim = document.getElementById('st_pmdeclim');
        
        if (pl_trandurh && st_pmralim && st_pmdeclim) {
            data.pl_trandurh = parseFloat(pl_trandurh.value) || 2.5; // Default transit duration
            data.st_pmralim = parseInt(st_pmralim.value) || 0; // Default to 0
            data.st_pmdeclim = parseInt(st_pmdeclim.value) || 0; // Default to 0
        } else {
            console.error('TESS fields not found in DOM');
            showToast('TESS fields not found. Please refresh the page.', 'error');
            return;
        }
    }
    
    console.log('Sending data for', currentDataset, ':', data);
    
    try {
        showLoading(true);
        setButtonLoading(elements.predictBtn, true);
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            // Provide more detailed error message for different error types
            let errorMessage = result.error || 'Prediction failed';
            if (errorMessage.includes('Feature shape mismatch') || errorMessage.includes('Model expects different features')) {
                errorMessage += '\n\nTip: The model was trained with different features. Try retraining the model or use a different dataset.';
            } else if (errorMessage.includes('Prediction failed')) {
                errorMessage += '\n\nTip: There was an issue with the model prediction. Try retraining the model or check your input data.';
            } else if (errorMessage.includes('Data preprocessing failed')) {
                errorMessage += '\n\nTip: Check that your input values are valid numbers within reasonable ranges.';
            }
            throw new Error(errorMessage);
        }
        
        displayManualResults(result);
        
        if (result.status === 'mock_prediction') {
            showToast('Demo prediction completed! (Using mock data - models not found)', 'warning');
        } else {
            showToast('Prediction completed successfully!', 'success');
        }
        
    } catch (error) {
        handleError(error, 'manual prediction');
    } finally {
        showLoading(false);
        setButtonLoading(elements.predictBtn, false);
    }
}

/**
 * Display manual prediction results
 */
function displayManualResults(result) {
    elements.manualResults.classList.remove('hidden');
    
    document.getElementById('prediction-class').textContent = result.prediction;
    document.getElementById('prediction-confidence').textContent = `${(result.confidence * 100).toFixed(1)}%`;
    
    const details = document.getElementById('prediction-details');
    details.innerHTML = `
        <div class="space-y-1">
            <div><strong>Dataset:</strong> ${result.dataset}</div>
            <div><strong>Features:</strong> Orbital Period: ${result.features.pl_orbper}, Transit Depth: ${result.features.pl_trandep}, Stellar Temperature: ${result.features.st_teff}</div>
            <div><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</div>
        </div>
    `;
}

/**
 * Handle file upload form submission
 */
async function handleFileUpload(event) {
    event.preventDefault();
    
    if (isLoading) return;
    
    const file = elements.csvFile.files[0];
    if (!file) {
        showToast('Please select a file first', 'error');
        return;
    }
    
    // Debug logging
    console.log('Upload attempt:', {
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type,
        dataset: currentDataset,
        retrain: elements.retrainCheckbox.checked
    });
    
    // Validate dataset
    if (!SUPPORTED_DATASETS.includes(currentDataset)) {
        showToast(`Invalid dataset: ${currentDataset}. Please select a valid dataset.`, 'error');
        return;
    }
    
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('Please select a CSV file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('dataset', currentDataset);
    formData.append('retrain', elements.retrainCheckbox.checked);
    
    try {
        showLoading(true);
        setButtonLoading(elements.uploadBtn, true);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        console.log('Upload response:', {
            status: response.status,
            statusText: response.statusText,
            headers: Object.fromEntries(response.headers.entries())
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            console.error('Upload failed:', result);
            // Provide more detailed error message for different error types
            let errorMessage = result.error || 'Upload failed';
            if (errorMessage.includes('Missing required columns') || (errorMessage.includes('Raw') && errorMessage.includes('dataset detected'))) {
                // Handle multi-line error messages with better formatting
                // Try both escaped and unescaped newlines
                let errorLines = errorMessage.split('\\n\\n');
                if (errorLines.length === 1) {
                    errorLines = errorMessage.split('\n\n');
                }
                
                if (errorLines.length > 1) {
                    // Show each part of the error message separately
                    errorLines.forEach((line, index) => {
                        if (line.trim()) {
                            const toastType = index === 0 ? 'error' : 'info';
                            showToast(line.trim(), toastType);
                        }
                    });
                    return; // Don't show the original error message
                }
                errorMessage += '\n\nTip: Download the CSV template to see the correct format.';
            } else if (errorMessage.includes('Data preprocessing failed')) {
                errorMessage += '\n\nTip: Ensure your CSV contains valid numeric data with positive orbital periods and reasonable stellar temperatures.';
            } else if (errorMessage.includes('must be positive') || errorMessage.includes('must be non-negative')) {
                errorMessage += '\n\nTip: Check that your data contains valid values - orbital periods must be positive, transit depths must be non-negative.';
            } else if (errorMessage.includes('Feature shape mismatch') || errorMessage.includes('Model expects different features')) {
                errorMessage += '\n\nTip: The model was trained with different features. Try retraining the model or use a different dataset.';
            } else if (errorMessage.includes('Prediction failed')) {
                errorMessage += '\n\nTip: There was an issue with the model prediction. Try retraining the model or check your input data.';
            }
            throw new Error(errorMessage);
        }
        
        if (elements.retrainCheckbox.checked) {
            displayRetrainResults(result);
        } else {
            displayUploadResults(result);
        }
        
        if (result.status === 'mock_predictions') {
            showToast('Demo processing completed! (Using mock data - models not found)', 'warning');
        } else {
            showToast(result.message || 'File processed successfully!', 'success');
        }
        
    } catch (error) {
        handleError(error, 'file upload');
    } finally {
        showLoading(false);
        setButtonLoading(elements.uploadBtn, false);
    }
}

/**
 * Display upload results
 */
function displayUploadResults(result) {
    elements.uploadResults.classList.remove('hidden');
    
    const summary = document.getElementById('upload-summary');
    summary.innerHTML = `
        <div class="space-y-2">
            <div><strong>Dataset:</strong> ${result.dataset}</div>
            <div><strong>Total Rows Processed:</strong> ${result.total_rows}</div>
            <div><strong>Status:</strong> Classification completed</div>
        </div>
    `;
    
    const tbody = document.getElementById('upload-results-body');
    tbody.innerHTML = '';
    
    result.results.slice(0, 10).forEach(row => { // Show first 10 results
        const tr = document.createElement('tr');
        const confidenceClass = getConfidenceClass(row.confidence);
        
        tr.innerHTML = `
            <td class="px-3 py-2">${row.row}</td>
            <td class="px-3 py-2 font-semibold">${row.prediction}</td>
            <td class="px-3 py-2">
                <span class="${confidenceClass} px-2 py-1 rounded-full text-xs font-medium border">
                    ${(row.confidence * 100).toFixed(1)}%
                </span>
            </td>
        `;
        tbody.appendChild(tr);
    });
    
    if (result.results.length > 10) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td colspan="3" class="px-3 py-2 text-center text-gray-400">
                Showing first 10 results of ${result.results.length} total
            </td>
        `;
        tbody.appendChild(tr);
    }
}

/**
 * Handle model retraining
 */
async function handleModelRetrain(event) {
    event.preventDefault();
    
    if (isLoading) return;
    
    const data = {
        dataset: currentDataset,
        n_estimators: parseInt(elements.nEstimatorsSlider.value),
        max_depth: parseInt(elements.maxDepthSlider.value),
        learning_rate: parseFloat(elements.learningRateSlider.value)
    };
    
    try {
        showLoading(true);
        setButtonLoading(elements.retrainBtn, true);
        
        const response = await fetch('/retrain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Retraining failed');
        }
        
        displayRetrainResults(result);
        showToast('Model retraining completed successfully!', 'success');
        
    } catch (error) {
        handleError(error, 'model retraining');
    } finally {
        showLoading(false);
        setButtonLoading(elements.retrainBtn, false);
    }
}

/**
 * Display retraining results
 */
function displayRetrainResults(result) {
    elements.retrainResults.classList.remove('hidden');
    elements.noRetrain.classList.add('hidden');
    
    const details = document.getElementById('retrain-details');
    details.innerHTML = `
        <div class="space-y-3">
            <div class="flex justify-between">
                <span class="text-purple-200">Dataset:</span>
                <span class="text-white font-semibold">${result.dataset}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-purple-200">New Accuracy:</span>
                <span class="text-green-400 font-bold text-xl">${(result.new_accuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="flex justify-between">
                <span class="text-purple-200">N Estimators:</span>
                <span class="text-white">${result.hyperparameters?.n_estimators || 'N/A'}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-purple-200">Max Depth:</span>
                <span class="text-white">${result.hyperparameters?.max_depth || 'N/A'}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-purple-200">Learning Rate:</span>
                <span class="text-white">${result.hyperparameters?.learning_rate || 'N/A'}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-purple-200">Training Samples:</span>
                <span class="text-white">${result.training_samples || 'N/A'}</span>
            </div>
        </div>
    `;
    
    // Reload statistics to reflect new accuracy
    loadStatistics();
}

/**
 * Load and display statistics
 */
async function loadStatistics() {
    try {
        const response = await fetch(`/stats?dataset=${currentDataset}`);
        const stats = await response.json();
        
        if (response.ok) {
            updateStatisticsDisplay(stats);
            updateClassDistributionChart(stats.class_distribution);
            
            // Show notification if using mock data
            if (stats.status === 'mock_data') {
                showToast('Using demo data - models not found. Please ensure model files are in the correct directory.', 'warning', 8000);
            }
        } else {
            console.error('Failed to load stats:', stats.error);
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

/**
 * Update statistics display
 */
function updateStatisticsDisplay(stats) {
    elements.accuracyValue.textContent = `${(stats.accuracy * 100).toFixed(0)}%`;
    elements.totalSamples.textContent = stats.total_samples?.toLocaleString() || 'N/A';
    elements.modelType.textContent = stats.model_type || 'XGBoost';
}

/**
 * Update class distribution chart
 */
function updateClassDistributionChart(classDistribution) {
    const ctx = document.getElementById('class-distribution-chart');
    
    // Destroy existing chart if it exists
    if (classDistributionChart) {
        classDistributionChart.destroy();
    }
    
    const labels = Object.keys(classDistribution);
    const data = Object.values(classDistribution);
    const colors = ['#FC3D21', '#0B3D91', '#22C55E', '#F59E0B', '#8B5CF6'];
    
    classDistributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors.slice(0, labels.length),
                borderColor: '#1a1a2e',
                borderWidth: 2,
                hoverOffset: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#e5e7eb',
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 26, 46, 0.9)',
                    titleColor: '#e5e7eb',
                    bodyColor: '#e5e7eb',
                    borderColor: '#FC3D21',
                    borderWidth: 1
                }
            }
        }
    });
}

/**
 * Input validation
 */
function validateInput(event) {
    const input = event.target;
    const value = parseFloat(input.value);
    
    clearValidation(event);
    
    if (isNaN(value) || value <= 0) {
        showInputError(input, 'Please enter a valid positive number');
        return false;
    }
    
    // Specific validation for each field
    switch (input.name) {
        case 'pl_orbper':
            if (value < 0.001 || value > 10000) {
                showInputError(input, 'Orbital period should be between 0.001 and 10000 days');
                return false;
            }
            break;
        case 'pl_trandep':
            if (value < 0 || value > 100000) {
                showInputError(input, 'Transit depth should be between 0 and 100000 ppm');
                return false;
            }
            break;
        case 'st_teff':
            if (value < 1000 || value > 100000) {
                showInputError(input, 'Stellar temperature should be between 1000 and 100000 K');
                return false;
            }
            break;
    }
    
    showInputSuccess(input);
    return true;
}

/**
 * Show input error
 */
function showInputError(input, message) {
    input.classList.add('form-error');
    
    const existingError = input.parentNode.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    input.parentNode.appendChild(errorDiv);
}

/**
 * Show input success
 */
function showInputSuccess(input) {
    input.classList.remove('form-error');
    input.classList.add('form-success');
    
    const existingError = input.parentNode.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
}

/**
 * Clear input validation
 */
function clearValidation(event) {
    const input = event.target;
    input.classList.remove('form-error', 'form-success');
    
    const errorMessage = input.parentNode.querySelector('.error-message');
    if (errorMessage) {
        errorMessage.remove();
    }
}

/**
 * Handle retrain checkbox change
 */
function handleRetrainCheckboxChange() {
    const isChecked = elements.retrainCheckbox.checked;
    const uploadText = elements.uploadBtn.querySelector('#upload-btn-text');
    
    if (isChecked) {
        uploadText.textContent = 'Upload & Retrain';
    } else {
        uploadText.textContent = 'Upload & Process';
    }
}

/**
 * Get confidence class for styling
 */
function getConfidenceClass(confidence) {
    if (confidence > 0.8) {
        return 'confidence-high';
    } else if (confidence > 0.6) {
        return 'confidence-medium';
    } else {
        return 'confidence-low';
    }
}

/**
 * Show/hide loading overlay
 */
function showLoading(show) {
    isLoading = show;
    if (show) {
        elements.loadingOverlay.classList.remove('hidden');
    } else {
        elements.loadingOverlay.classList.add('hidden');
    }
}

/**
 * Set button loading state
 */
function setButtonLoading(button, loading) {
    const text = button.querySelector('span');
    const spinner = button.querySelector('i.fa-spinner');
    
    if (loading) {
        button.disabled = true;
        button.classList.add('opacity-50', 'cursor-not-allowed');
        if (spinner) spinner.classList.remove('hidden');
    } else {
        button.disabled = false;
        button.classList.remove('opacity-50', 'cursor-not-allowed');
        if (spinner) spinner.classList.add('hidden');
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info', duration = 5000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = getToastIcon(type);
    toast.innerHTML = `
        <div class="flex items-center">
            <i class="${icon} mr-3"></i>
            <span>${message}</span>
        </div>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    // Auto remove after duration
    setTimeout(() => {
        toast.classList.add('slide-out');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, duration);
}

/**
 * Get toast icon based on type
 */
function getToastIcon(type) {
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    };
    return icons[type] || icons.info;
}

/**
 * Utility function to format numbers
 */
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

/**
 * Utility function to debounce function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Initialize accessibility features
 */
function initializeAccessibility() {
    // Set up reduced motion preferences
    if (userPreferences.reducedMotion) {
        document.documentElement.style.setProperty('--animation-duration', '0.01ms');
        document.documentElement.style.setProperty('--transition-duration', '0.01ms');
    }
    
    // Set up high contrast mode if enabled
    if (userPreferences.highContrast) {
        toggleHighContrast();
    }
    
    // Add focus indicators for keyboard navigation
    addFocusIndicators();
}

/**
 * Toggle high contrast mode
 */
function toggleHighContrast() {
    highContrastMode = !highContrastMode;
    userPreferences.highContrast = highContrastMode;
    
    if (highContrastMode) {
        document.body.classList.add('high-contrast');
        elements.highContrastToggle.innerHTML = '<i class="fas fa-adjust mr-2" aria-hidden="true"></i><span class="text-sm font-medium">Normal Contrast</span>';
        announceToScreenReader('High contrast mode enabled');
    } else {
        document.body.classList.remove('high-contrast');
        elements.highContrastToggle.innerHTML = '<i class="fas fa-adjust mr-2" aria-hidden="true"></i><span class="text-sm font-medium">High Contrast</span>';
        announceToScreenReader('High contrast mode disabled');
    }
    
    // Save preference
    saveUserPreferences();
}

/**
 * Handle keyboard navigation
 */
function handleKeyboardNavigation(event) {
    // ESC key to close modals or clear focus
    if (event.key === 'Escape') {
        const activeElement = document.activeElement;
        if (activeElement && activeElement.blur) {
            activeElement.blur();
        }
    }
    
    // Tab navigation enhancement
    if (event.key === 'Tab') {
        document.body.classList.add('keyboard-navigation');
    }
}

/**
 * Add focus indicators for keyboard navigation
 */
function addFocusIndicators() {
    const style = document.createElement('style');
    style.textContent = `
        .keyboard-navigation *:focus {
            outline: 2px solid var(--nasa-red) !important;
            outline-offset: 2px !important;
        }
    `;
    document.head.appendChild(style);
}

/**
 * Announce message to screen readers
 */
function announceToScreenReader(message) {
    if (elements.liveRegion) {
        elements.liveRegion.textContent = message;
        // Clear after announcement
        setTimeout(() => {
            elements.liveRegion.textContent = '';
        }, 1000);
    }
}

/**
 * Load user preferences from localStorage
 */
function loadUserPreferences() {
    try {
        const saved = localStorage.getItem('nasa-exoplanet-preferences');
        if (saved) {
            const preferences = JSON.parse(saved);
            Object.assign(userPreferences, preferences);
        }
    } catch (error) {
        console.warn('Could not load user preferences:', error);
    }
}

/**
 * Save user preferences to localStorage
 */
function saveUserPreferences() {
    try {
        localStorage.setItem('nasa-exoplanet-preferences', JSON.stringify(userPreferences));
    } catch (error) {
        console.warn('Could not save user preferences:', error);
    }
}

/**
 * Enhanced error handling with accessibility
 */
function handleError(error, context = '') {
    console.error(`Error in ${context}:`, error);
    
    const errorMessage = error.message || 'An unexpected error occurred';
    showToast(errorMessage, 'error');
    announceToScreenReader(`Error: ${errorMessage}`);
}

/**
 * Enhanced success handling with accessibility
 */
function handleSuccess(message, context = '') {
    console.log(`Success in ${context}:`, message);
    showToast(message, 'success');
    announceToScreenReader(`Success: ${message}`);
}

/**
 * Enhanced loading state management
 */
function setLoadingState(loading, context = '') {
    isLoading = loading;
    
    if (loading) {
        announceToScreenReader('Processing request, please wait');
    } else {
        announceToScreenReader('Processing complete');
    }
}

// Export functions for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateInput,
        getConfidenceClass,
        formatNumber,
        debounce,
        toggleHighContrast,
        announceToScreenReader,
        handleError,
        handleSuccess
    };
}