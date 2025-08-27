const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');
const { v4: uuidv4 } = require('uuid');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 5050;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Create directories function
const createDirectories = async () => {
    const dirs = ['uploads', 'results', 'pdb_files', 'public/css', 'public/js', 'public/images', 'views'];
    for (const dir of dirs) {
        try {
            await fs.mkdir(dir, { recursive: true });
        } catch (error) {
            // Directory already exists
        }
    }
};

// File upload configuration
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, 'uploads/'),
    filename: (req, file, cb) => cb(null, `${uuidv4()}_${file.originalname}`)
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 16 * 1024 * 1024 },
    fileFilter: (req, file, cb) => {
        const allowedTypes = ['.fasta', '.fa', '.txt'];
        const ext = path.extname(file.originalname).toLowerCase();
        cb(null, allowedTypes.includes(ext));
    }
});

// Job tracking
const jobStatus = new Map();
const jobResults = new Map();

class JobManager {
    static createJob(jobId, sequence) {
        const job = {
            job_id: jobId,
            sequence,
            status: 'queued',
            progress: 0,
            stage: 'Initializing analysis...',
            start_time: new Date()
        };
        jobStatus.set(jobId, job);
        return job;
    }

    static updateJob(jobId, updates) {
        const job = jobStatus.get(jobId);
        if (job) Object.assign(job, updates);
        return job;
    }

    static getJob(jobId) {
        return jobStatus.get(jobId);
    }
}

// Python workflow integration using JSON interface
const callPythonWorkflow = async (sequence, jobId) => {
    return new Promise((resolve, reject) => {
        console.log(`Starting Python workflow for job ${jobId}`);
        
        // Check if workflow_json_processor.py exists
        if (!fsSync.existsSync('workflow_json_processor.py')) {
            console.log('workflow_json_processor.py not found, checking final_workflow4.py...');
            
            if (!fsSync.existsSync('final_workflow4.py')) {
                const error = 'Python workflow script not found. Please ensure workflow_json_processor.py or final_workflow4.py exists in the server directory.';
                console.error(error);
                reject(new Error(error));
                return;
            }
            
            // Fallback to direct final_workflow4.py call
            return callDirectPythonWorkflow(sequence, jobId, resolve, reject);
        }

        // Prepare input data for JSON processor
        const inputData = {
            sequence: sequence,
            job_id: jobId,
            timestamp: new Date().toISOString()
        };

        const inputJson = JSON.stringify(inputData);
        
        // Spawn Python process with JSON interface
        const pythonProcess = spawn('python3', ['workflow_json_processor.py', inputJson], {
            stdio: ['pipe', 'pipe', 'pipe'],
            cwd: process.cwd()
        });

        let stdout = '';
        let stderr = '';

        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
            console.log('Python stderr:', data.toString());
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                try {
                    // Parse JSON output
                    const result = JSON.parse(stdout.trim());
                    
                    if (result.success) {
                        console.log('Python workflow completed successfully');
                        resolve(result);
                    } else {
                        const error = `Python workflow failed: ${result.error || 'Unknown error'}`;
                        console.error(error);
                        reject(new Error(error));
                    }
                } catch (error) {
                    console.error('Failed to parse Python JSON output:', error.message);
                    console.log('Python output:', stdout);
                    reject(new Error(`Invalid JSON output from Python workflow: ${error.message}`));
                }
            } else {
                console.error('Python process failed with code:', code);
                console.error('Python stderr:', stderr);
                reject(new Error(`Python process exited with code ${code}: ${stderr || 'No error details'}`));
            }
        });

        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error.message);
            reject(new Error(`Failed to start Python workflow: ${error.message}`));
        });
    });
};

// Fallback for direct final_workflow4.py call
const callDirectPythonWorkflow = (sequence, jobId, resolve, reject) => {
    console.log('Using direct final_workflow4.py call');
    
    const tempScript = `
import sys
import json
from final_workflow4 import run_final_workflow

def main():
    sequence = "${sequence.replace(/"/g, '\\"')}"
    try:
        result = run_final_workflow(sequence)
        if result:
            # Convert result to JSON-serializable format
            json_result = {"success": True, "job_id": "${jobId}"}
            for key, value in result.items():
                if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    json_result[key] = value
                else:
                    json_result[key] = str(value)
            print(json.dumps(json_result))
        else:
            print(json.dumps({"success": False, "error": "No result from workflow", "job_id": "${jobId}"}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e), "job_id": "${jobId}"}))

if __name__ == "__main__":
    main()
`;

    const tempScriptPath = `temp_workflow_${jobId}.py`;
    fsSync.writeFileSync(tempScriptPath, tempScript);

    const pythonProcess = spawn('python3', [tempScriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
        // Cleanup temp script
        try {
            fsSync.unlinkSync(tempScriptPath);
        } catch (e) {
            console.log('Failed to cleanup temp script:', e.message);
        }

        if (code === 0) {
            try {
                const lastLine = stdout.trim().split('\n').pop();
                const result = JSON.parse(lastLine);
                
                if (result.success) {
                    console.log('Direct Python workflow completed successfully');
                    resolve(result);
                } else {
                    console.error('Direct Python workflow error:', result.error);
                    reject(new Error(`Python workflow failed: ${result.error}`));
                }
            } catch (error) {
                console.error('Failed to parse direct Python output');
                reject(new Error(`Failed to parse Python output: ${error.message}`));
            }
        } else {
            console.error('Direct Python process failed with code:', code);
            reject(new Error(`Python process failed with code ${code}: ${stderr || stdout}`));
        }
    });

    pythonProcess.on('error', (error) => {
        reject(new Error(`Failed to start Python process: ${error.message}`));
    });
};

const processResult = (result, jobId, sequence) => {
    // If result already has web display format, use it
    if (result.aligned_display && result.metadata_text) {
        return result;
    }
    
    // Otherwise, format it for web display
    const structure = result['combined structure'] || result.structure;
    const mask = result['structure mask'] || Array(sequence.length).fill('M');
    
    // Calculate coverage
    const coverage = result.dssp ? (result.dssp.length / sequence.length) * 100 : 0;

    // Create metadata text
    const disorder = result.disorder || {};
    const topHit = result['top hit'] || {};
    
    const metadataText = `PROTEIN CLASSIFICATION: ${result.classification || 'Unknown'}
DISORDER FRACTION: ${disorder.disorder_fraction || '0.00'}
DISORDERED REGIONS: ${disorder.regions || 'None detected'}
MEMBRANE: ${result.membrane || 'Non-membrane'}
TRANSMEMBRANE REGIONS: ${Array.isArray(result.tm_regions) ? result.tm_regions.join(', ') || 'None detected' : 'None detected'}
TOP BLAST HIT: ${topHit.subject_id || 'Unknown'} | PERCENT IDENTITY: ${(topHit.percent_identity || 0).toFixed(2)}% | E-VALUE: ${(topHit.evalue || 1.0).toExponential(2)} | COVERAGE: ${coverage.toFixed(2)}%
ACCESSION: ${result.accession || 'Unknown'}
MODEL USED: ${result.model_used || 'Unknown'}
TIME: ${result.time || 'Unknown'}`;

    // Create aligned display
    const alignedDisplay = [];
    const blockSize = 60;

    for (let i = 0; i < sequence.length; i += blockSize) {
        const seqChunk = sequence.slice(i, i + blockSize);
        const strChunk = structure.slice(i, i + blockSize);
        const maskChunk = mask.slice(i, i + blockSize);

        const structureColored = strChunk.split('').map((char, idx) => ({
            char,
            source: maskChunk[idx] || 'M',
            class: getStructureClass(char, maskChunk[idx] || 'M')
        }));

        const sequenceColored = seqChunk.split('').map((char, idx) => ({
            char,
            source: maskChunk[idx] || 'M',
            class: getAminoAcidClass(maskChunk[idx] || 'M')
        }));

        alignedDisplay.push({
            position: i + 1,
            sequence: seqChunk,
            structure: strChunk,
            structure_colored: structureColored,
            sequence_colored: sequenceColored
        });
    }

    return {
        ...result,
        aligned_display: alignedDisplay,
        metadata_text: metadataText,
        coverage,
        timestamp: result.timestamp || new Date().toISOString()
    };
};

// Helper functions for CSS classes
const getStructureClass = (char, source) => {
    const helixChars = ['H', 'G', 'I'];
    const sheetChars = ['E', 'B'];
    const coilChars = ['C', 'T', 'S', '-'];

    let structType = 'unknown';
    if (helixChars.includes(char)) structType = 'helix';
    else if (sheetChars.includes(char)) structType = 'sheet';
    else if (coilChars.includes(char)) structType = 'coil';

    if (source === 'D') return `structure-${structType} source-dssp`;
    else if (source === 'M') return `structure-${structType} source-model`;
    else if (source === 'S') return `structure-${structType} source-smoothed`;
    else return `structure-${structType}`;
};

const getAminoAcidClass = (source) => {
    return source === 'D' ? 'dssp-amino-acid' : '';
};

// Enhanced workflow with progress tracking
const runWorkflow = async (jobId, sequence) => {
    const stages = [
        { progress: 5, stage: 'Initializing analysis...' },
        { progress: 15, stage: 'Loading ProtBERT model...' },
        { progress: 25, stage: 'Processing protein sequence...' },
        { progress: 40, stage: 'Predicting secondary structure...' },
        { progress: 65, stage: 'Analyzing structure data...' },
        { progress: 80, stage: 'Generating visualization...' },
        { progress: 95, stage: 'Finalizing results...' }
    ];

    // Update progress through stages
    for (let i = 0; i < stages.length - 1; i++) {
        await new Promise(resolve => setTimeout(resolve, 500));
        JobManager.updateJob(jobId, { ...stages[i], status: 'running' });
    }

    try {
        // Call Python workflow - will throw error if it fails
        const result = await callPythonWorkflow(sequence, jobId);
        
        // Process result for web display
        const processedResult = processResult(result, jobId, sequence);
        
        // Store results
        jobResults.set(jobId, processedResult);
        
        // Update job status
        JobManager.updateJob(jobId, { 
            status: 'completed', 
            progress: 100, 
            stage: 'Analysis complete!',
            end_time: new Date(),
            result: processedResult 
        });
        
    } catch (error) {
        console.error('Workflow error:', error);
        
        // Store error details
        const errorMessage = error.message || 'Unknown error occurred';
        
        JobManager.updateJob(jobId, { 
            status: 'failed', 
            progress: 100, 
            stage: 'Analysis failed',
            end_time: new Date(),
            error: errorMessage 
        });
        
        // Also store in results for error display
        jobResults.set(jobId, {
            success: false,
            error: errorMessage,
            job_id: jobId,
            timestamp: new Date().toISOString()
        });
    }
};

// Routes
app.get('/', (req, res) => res.render('index'));

app.post('/predict', upload.single('fasta_file'), async (req, res) => {
    try {
        let sequence = req.body.sequence?.trim() || '';
        let accession = null;

        // Handle file upload
        if (req.file) {
            const content = await fs.readFile(req.file.path, 'utf8');
            if (content.startsWith('>')) {
                const lines = content.split('\n');
                accession = lines[0].substring(1).split(' ')[0];
                sequence = lines.slice(1).join('').replace(/\s/g, '').toUpperCase();
            } else {
                sequence = content.replace(/\s/g, '').toUpperCase();
            }
            await fs.unlink(req.file.path);
        }

        // Handle pasted FASTA content
        if (sequence.startsWith('>')) {
            const lines = sequence.split('\n');
            accession = lines[0].substring(1).split(' ')[0];
            sequence = lines.slice(1).join('').replace(/\s/g, '').toUpperCase();
        }

        // Validation
        if (!sequence || sequence.length < 10) {
            return res.status(400).json({ error: 'Sequence must be at least 10 amino acids long' });
        }

        const validAA = /^[ACDEFGHIKLMNPQRSTVWY]*$/i;
        if (!validAA.test(sequence)) {
            return res.status(400).json({ error: 'Sequence contains invalid amino acid characters' });
        }

        if (sequence.length > 5000) {
            return res.status(400).json({ error: 'Sequence too long (maximum 5000 amino acids)' });
        }

        const jobId = uuidv4();
        JobManager.createJob(jobId, sequence);
        
        // Start workflow asynchronously
        runWorkflow(jobId, sequence);

        res.json({ job_id: jobId, status: 'queued' });
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/status/:jobId', (req, res) => {
    const job = JobManager.getJob(req.params.jobId);
    if (!job) return res.status(404).json({ error: 'Job not found' });
    
    const statusData = {
        job_id: job.job_id,
        status: job.status,
        progress: job.progress,
        stage: job.stage,
        start_time: job.start_time.toISOString(),
        end_time: job.end_time ? job.end_time.toISOString() : null,
        error: job.error
    };
    
    res.json(statusData);
});

app.get('/results/:jobId', (req, res) => {
    const result = jobResults.get(req.params.jobId);
    if (!result) return res.status(404).json({ error: 'Results not found' });
    res.json(result);
});

app.get('/results_page/:jobId', (req, res) => {
    res.render('results', { job_id: req.params.jobId });
});

app.get('/download/:jobId/:format', (req, res) => {
    const result = jobResults.get(req.params.jobId);
    if (!result) return res.status(404).json({ error: 'Results not found' });

    const { format } = req.params;
    let content, filename;

    if (format === 'fasta') {
        content = `>Prediction_${req.params.jobId} | ${result.accession || 'Unknown'} | Model: ${result.model_used || 'Unknown'}
${result.sequence}
>Secondary_Structure_${req.params.jobId}
${result.structure || result['combined structure']}`;
        filename = `prediction_${req.params.jobId}.fasta`;
    } else {
        const blockSize = 60;
        let textContent = `ProtBERT Protein Secondary Structure Prediction Results
${'='.repeat(60)}

Job ID: ${req.params.jobId}
Timestamp: ${result.timestamp}
${result.metadata_text || 'No metadata available'}

Sequence and Structure Alignment:
${'-'.repeat(60)}

`;
        
        const sequence = result.sequence || '';
        const structure = result.structure || result['combined structure'] || '';
        
        for (let i = 0; i < sequence.length; i += blockSize) {
            const seqChunk = sequence.substring(i, i + blockSize);
            const strChunk = structure.substring(i, i + blockSize);
            textContent += `SEQ ${(i + 1).toString().padStart(4, ' ')}: ${seqChunk}\n`;
            textContent += `STR ${(i + 1).toString().padStart(4, ' ')}: ${strChunk}\n\n`;
        }
        
        content = textContent;
        filename = `prediction_${req.params.jobId}.txt`;
    }

    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.setHeader('Content-Type', format === 'fasta' ? 'text/plain' : 'text/plain');
    res.send(content);
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
});

// Test endpoint for development
app.get('/test/:sequence', async (req, res) => {
    try {
        const sequence = req.params.sequence.toUpperCase();
        const jobId = 'test_' + Date.now();
        
        console.log(`Test endpoint called with sequence: ${sequence}`);
        
        const result = await callPythonWorkflow(sequence, jobId);
        const processedResult = processResult(result, jobId, sequence);
        
        res.json(processedResult);
    } catch (error) {
        console.error('Test endpoint error:', error);
        res.status(500).json({ 
            success: false,
            error: error.message,
            details: 'The Python workflow failed. Please check that the Python environment is properly configured and all dependencies are installed.'
        });
    }
});

app.get('/about', (req, res) => res.render('about'));
app.get('/help', (req, res) => res.render('help'));
app.use((req, res) => res.status(404).render('404'));

// Start server
createDirectories().then(() => {
    app.listen(PORT, () => {
        console.log(`🚀 SSTRAND server running on http://localhost:${PORT}`);
        console.log('📊 Features available:');
        console.log('  • Single sequence prediction');
        console.log('  • FASTA file upload');
        console.log('  • Real-time progress tracking');
        console.log('  • Results download (FASTA/TXT)');
        console.log('  • JSON API integration');
        
        // Check if Python workflow is available
        if (fsSync.existsSync('workflow_json_processor.py')) {
            console.log('✅ JSON workflow processor detected');
        } else if (fsSync.existsSync('final_workflow4.py')) {
            console.log('✅ Python workflow detected: final_workflow4.py');
        } else {
            console.log('⚠️ Python workflow not found - analysis will fail with error message');
        }
        
        // Test endpoints
        console.log('🔗 Available endpoints:');
        console.log(`  • Main interface: http://localhost:${PORT}/`);
        console.log(`  • Health check: http://localhost:${PORT}/health`);
        console.log(`  • Test sequence: http://localhost:${PORT}/test/MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN`);
    });
});