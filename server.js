import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { pipeline, env } from '@huggingface/transformers';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// Configure Transformers.js environment
// Optional: specify custom cache location (defaults to OS user cache if unset)
if (process.env.HF_CACHE_DIR) {
  env.cacheDir = path.resolve(process.env.HF_CACHE_DIR);
}

// Global pipeline reference
let classifier = null;
let isModelLoading = false;
let modelLoadError = null;

/**
 * Initialize the Horizon-Labs phishing scam detector model.
 * Loads the quantized (q8) ONNX weights for fast, low-memory inference.
 */
async function initModel() {
  if (classifier) return classifier;
  if (isModelLoading) return;

  isModelLoading = true;
  modelLoadError = null;

  console.log('[Model] Starting model download/load: Horizon-Labs/phishing-scam-detector-base (dtype: q8)...');
  const startTime = Date.now();

  try {
    classifier = await pipeline(
      'text-classification',
      'Horizon-Labs/phishing-scam-detector-base',
      { dtype: 'q8' }
    );
    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`[Model] Loaded successfully in ${duration}s.`);
    return classifier;
  } catch (err) {
    modelLoadError = err.message || 'Unknown error during pipeline initialization';
    console.error('[Model] Failed to load pipeline:', err);
    throw err;
  } finally {
    isModelLoading = false;
  }
}

/**
 * Format raw pipeline output array into structured classification metrics.
 */
function processModelOutput(results, text, threshold = 0.5) {
  const scores = results.reduce((acc, curr) => {
    acc[curr.label.toLowerCase()] = curr.score;
    return acc;
  }, {});

  const fraudScore = scores['fraud'] ?? 0;
  const spamScore = scores['spam'] ?? 0;
  const legitimateScore = scores['legitimate'] ?? 0;
  const unwantedScore = fraudScore + spamScore;

  const isSafe = fraudScore < threshold && unwantedScore < threshold;

  let verdict = 'legitimate';
  if (fraudScore >= threshold) {
    verdict = 'fraud';
  } else if (spamScore >= threshold) {
    verdict = 'spam';
  }

  const riskLevel = fraudScore > 0.7 ? 'HIGH' : fraudScore > 0.4 ? 'MEDIUM' : 'LOW';

  return {
    input: text,
    verdict,
    is_safe: isSafe,
    risk_level: riskLevel,
    scores: {
      fraud: Number(fraudScore.toFixed(4)),
      spam: Number(spamScore.toFixed(4)),
      legitimate: Number(legitimateScore.toFixed(4)),
      unwanted_total: Number(unwantedScore.toFixed(4)),
    },
  };
}

// -----------------------------------------------------------------------------
// Routes
// -----------------------------------------------------------------------------

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(classifier ? 200 : 503).json({
    status: classifier ? 'ready' : isModelLoading ? 'loading' : 'offline',
    model: 'Horizon-Labs/phishing-scam-detector-base',
    error: modelLoadError,
  });
});

// Single message classification
app.post('/api/analyze', async (req, res) => {
  try {
    if (!classifier) {
      return res.status(503).json({
        error: 'Model is not initialized yet. Check /health status.',
        details: modelLoadError,
      });
    }

    const { text, threshold = 0.5 } = req.body;

    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      return res.status(400).json({ error: 'A valid, non-empty "text" string is required.' });
    }

    const cleanText = text.trim();
    const rawResults = await classifier(cleanText, { top_k: null });
    const payload = processModelOutput(rawResults, cleanText, Number(threshold));

    return res.json({
      success: true,
      ...payload,
    });
  } catch (error) {
    console.error('Inference error (/api/analyze):', error);
    return res.status(500).json({
      error: 'Internal inference error',
      details: error.message,
    });
  }
});

// Batch classification
app.post('/api/analyze-batch', async (req, res) => {
  try {
    if (!classifier) {
      return res.status(503).json({
        error: 'Model is not initialized yet. Check /health status.',
      });
    }

    const { texts, threshold = 0.5 } = req.body;

    if (!Array.isArray(texts) || texts.length === 0) {
      return res.status(400).json({ error: '"texts" must be a non-empty array of strings.' });
    }

    // Limit batch size to prevent event-loop choking
    const batch = texts.slice(0, 50).map((t) => (typeof t === 'string' ? t.trim() : ''));
    const rawResults = await classifier(batch, { top_k: null });

    const results = rawResults.map((result, idx) =>
      processModelOutput(result, batch[idx], Number(threshold))
    );

    return res.json({
      success: true,
      count: results.length,
      data: results,
    });
  } catch (error) {
    console.error('Inference error (/api/analyze-batch):', error);
    return res.status(500).json({
      error: 'Internal inference error',
      details: error.message,
    });
  }
});

// -----------------------------------------------------------------------------
// Server Start
// -----------------------------------------------------------------------------

async function startServer() {
  try {
    // Pre-warm the model before listening for traffic
    await initModel();

    app.listen(PORT, () => {
      console.log(`Server listening at http://localhost:${PORT}`);
      console.log(`Health check: http://localhost:${PORT}/health`);
      console.log(`Inference: POST http://localhost:${PORT}/api/analyze`);
    });
  } catch (err) {
    console.error('Fatal initialization error:', err);
    process.exit(1);
  }
}

startServer();