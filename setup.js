const fs = require('fs').promises;

const createTemplates = async () => {
    // Create views directory
    await fs.mkdir('views', { recursive: true });
    await fs.mkdir('public/css', { recursive: true });
    await fs.mkdir('public/js', { recursive: true });

    // Create index.ejs
    const indexTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SSTRAND - Protein Secondary Structure Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="stylesheet" href="/css/style.css">
</head>
<body>
    <div class="header-section">
        <div class="header-content">
            <div class="header-brand">
                <h1 class="header-title"><i class="fas fa-dna me-2"></i>SSTRAND</h1>
                <p class="header-subtitle">Protein Secondary Structure Prediction</p>
            </div>
            <div class="header-nav">
                <a href="/" class="nav-btn">HOME</a>
                <a href="/about" class="nav-btn">ABOUT</a>
                <a href="/help" class="nav-btn">CONTACT</a>
            </div>
        </div>
    </div>

    <main class="main-content">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h2>🧬 Start Your Analysis</h2>
                    <div class="card p-4">
                        <textarea id="sequenceInput" class="form-control mb-3" placeholder="Enter protein sequence..." rows="8"></textarea>
                        
                        <div class="upload-section mb-3" id="uploadSection">
                            <input type="file" id="fastaFile" accept=".fasta,.fa,.txt" style="display: none;">
                            <label for="fastaFile" class="btn btn-outline-primary w-100">
                                <i class="fas fa-upload me-2"></i>Upload FASTA File
                            </label>
                            <div id="fileInfo" class="mt-2" style="display: none;"></div>
                        </div>

                        <div class="d-flex gap-2">
                            <button class="btn btn-primary flex-fill" id="analyzeBtn">
                                <i class="fas fa-play me-2"></i>Analyze
                            </button>
                            <button class="btn btn-secondary" id="clearBtn">Clear</button>
                            <button class="btn btn-warning" id="exampleBtn">Example</button>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <h2>📖 About SSTRAND</h2>
                    <div class="card p-4">
                        <p>Advanced protein secondary structure prediction using ProtBERT and experimental data.</p>
                        <ul>
                            <li>🧬 Machine Learning Models</li>
                            <li>🔬 DSSP Integration</li>
                            <li>📊 Comprehensive Analysis</li>
                            <li>🎨 Visual Results</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="/js/main.js"></script>
</body>
</html>`;

    // Create results.ejs
    const resultsTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SSTRAND Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="stylesheet" href="/css/style.css">
</head>
<body>
    <div class="container mt-4">
        <h1><i class="fas fa-dna me-2"></i>Analysis Results</h1>
        
        <div id="loadingSection">
            <div class="text-center">
                <div class="spinner-border text-primary" role="status"></div>
                <p class="mt-2">Job ID: <strong><%= job_id %></strong></p>
                <p id="statusText">Analyzing...</p>
                <div class="progress mb-3">
                    <div class="progress-bar" id="progressBar" style="width: 0%"></div>
                </div>
            </div>
        </div>

        <div id="resultsSection" style="display: none;">
            <div class="card mb-3">
                <div class="card-header"><h5>Metadata</h5></div>
                <div class="card-body">
                    <pre id="metadataContent"></pre>
                </div>
            </div>

            <div class="card mb-3">
                <div class="card-header"><h5>Sequence Alignment</h5></div>
                <div class="card-body">
                    <div id="alignmentContent" style="font-family: monospace;"></div>
                </div>
            </div>

            <div class="text-center">
                <a href="/download/<%= job_id %>/fasta" class="btn btn-primary me-2">Download FASTA</a>
                <a href="/download/<%= job_id %>/txt" class="btn btn-secondary">Download Text</a>
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        const jobId = '<%= job_id %>';
        
        function pollStatus() {
            $.get('/status/' + jobId).done(function(status) {
                $('#progressBar').css('width', status.progress + '%');
                $('#statusText').text(status.stage);
                
                if (status.status === 'completed') {
                    loadResults();
                } else if (status.status !== 'failed') {
                    setTimeout(pollStatus, 1000);
                }
            });
        }

        function loadResults() {
            $.get('/results/' + jobId).done(function(data) {
                $('#loadingSection').hide();
                $('#metadataContent').text(data.metadata_text);
                
                let html = '';
                data.aligned_display.forEach(block => {
                    html += \`<div class="mb-2">
                        <div>SEQ \${block.position}: \${block.sequence}</div>
                        <div>STR \${block.position}: \${block.structure.split('').map(c => 
                            \`<span class="structure-\${c}">\${c}</span>\`).join('')}</div>
                    </div>\`;
                });
                $('#alignmentContent').html(html);
                $('#resultsSection').show();
            });
        }

        $(document).ready(function() {
            pollStatus();
        });
    </script>
</body>
</html>`;

    // Create other templates
    const aboutTemplate = `<!DOCTYPE html>
<html><head><title>About SSTRAND</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head><body>
<div class="container mt-4">
    <h1>About SSTRAND</h1>
    <p>Advanced protein secondary structure prediction tool.</p>
    <a href="/" class="btn btn-primary">Back to Home</a>
</div></body></html>`;

    const helpTemplate = `<!DOCTYPE html>
<html><head><title>Help - SSTRAND</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head><body>
<div class="container mt-4">
    <h1>Help & Contact</h1>
    <p>For support, contact: support@sstrand.com</p>
    <a href="/" class="btn btn-primary">Back to Home</a>
</div></body></html>`;

    const errorTemplate = `<!DOCTYPE html>
<html><head><title>Page Not Found</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head><body>
<div class="container mt-4 text-center">
    <h1>404 - Page Not Found</h1>
    <a href="/" class="btn btn-primary">Go Home</a>
</div></body></html>`;

    // Write all template files
    await fs.writeFile('views/index.ejs', indexTemplate);
    await fs.writeFile('views/results.ejs', resultsTemplate);
    await fs.writeFile('views/about.ejs', aboutTemplate);
    await fs.writeFile('views/help.ejs', helpTemplate);
    await fs.writeFile('views/404.ejs', errorTemplate);

    console.log('✅ All template files created');
};

createTemplates().catch(console.error);