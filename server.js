import express from 'express';
import cors from 'cors';
import { pipeline } from '@huggingface/transformers';

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS for all origins explicitly
app.use(cors({ origin: '*' }));
app.use(express.json());

let classifier = null;

// Initialize the model during boot
async function initModel() {
  console.log('Loading Horizon-Labs/phishing-scam-detector-base model...');
  
  // Using quantized q8 ONNX model for low memory footprint and high inference speed
  classifier = await pipeline(
    'text-classification',
    'Horizon-Labs/phishing-scam-detector-base',
    { dtype: 'q8' }
  );
  
  console.log('Model loaded and ready for inference.');
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: classifier ? 'ready' : 'loading',
    model: 'Horizon-Labs/phishing-scam-detector-base'
  });
});

// Primary classification endpoint
app.post('/api/analyze', async (req, res) => {
  try {
    if (!classifier) {
      return res.status(503).json({ error: 'Model pipeline is still loading. Try again in a few seconds.' });
    }

    const { text, threshold = 0.5 } = req.body;

    if (!text || typeof text !== 'string' || text.trim() === '') {
      return res.status(400).json({ error: 'A non-empty "text" string is required in the request body.' });
    }

    // Run inference and request all class scores
    const results = await classifier(text.trim(), { top_k: null });

    // Map output array [{ label: 'fraud', score: 0.92 }, ...] into an object
    const scores = results.reduce((acc, curr) => {
      acc[curr.label.toLowerCase()] = curr.score;
      return acc;
    }, {});

    const fraudScore = scores['fraud'] ?? 0;
    const spamScore = scores['spam'] ?? 0;
    const legitimateScore = scores['legitimate'] ?? 0;

    const unwantedScore = fraudScore + spamScore;
    
    // FIX: A message is safe ONLY IF both fraud and unwanted combined scores stay under the threshold
    const isSafe = (fraudScore < threshold) && (unwantedScore < threshold);

    return res.json({
      success: true,
      input: text,
      verdict: fraudScore >= threshold ? 'fraud' : (spamScore >= threshold ? 'spam' : 'legitimate'),
      is_safe: isSafe,
      risk_level: fraudScore > 0.7 ? 'HIGH' : fraudScore > 0.4 ? 'MEDIUM' : 'LOW',
      scores: {
        fraud: fraudScore,
        spam: spamScore,
        legitimate: legitimateScore,
        unwanted_total: unwantedScore
      }
    });
  } catch (error) {
    console.error('Inference error:', error);
    return res.status(500).json({ error: 'Internal server error during classification.' });
  }
});

// Start server after loading model
initModel()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`Scam detector backend running at http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    console.error('Failed to initialize model:', err);
    process.exit(1);
  });
