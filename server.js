import express from 'express';
import cors from 'cors';
import { pipeline } from '@huggingface/transformers';
import path from 'path';

const app = express();

app.use(cors({ origin: '*' }));
app.use(express.json());

let pipelinePromise = null;

function getClassifier() {
  if (!pipelinePromise) {
    
    process.env.HF_HOME = path.join('/tmp', '.hf-cache');
    
    console.log('Initializing Horizon-Labs/phishing-scam-detector-base model...');
    pipelinePromise = pipeline(
      'text-classification',
      'Horizon-Labs/phishing-scam-detector-base',
      { dtype: 'q8' }
    );
  }
  return pipelinePromise;
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'active',
    environment: 'serverless',
    model: 'Horizon-Labs/phishing-scam-detector-base'
  });
});

// Primary classification endpoint
app.post('/api/analyze', async (req, res) => {
  try {
    const { text, threshold = 0.5 } = req.body;

    if (!text || typeof text !== 'string' || text.trim() === '') {
      return res.status(400).json({ error: 'A non-empty "text" string is required in the request body.' });
    }

    // Safely retrieve or initialize the model instance
    const classifier = await getClassifier();
    const results = await classifier(text.trim(), { top_k: null });

    const scores = results.reduce((acc, curr) => {
      acc[curr.label.toLowerCase()] = curr.score;
      return acc;
    }, {});

    const fraudScore = scores['fraud'] ?? 0;
    const spamScore = scores['spam'] ?? 0;
    const legitimateScore = scores['legitimate'] ?? 0;
    const unwantedScore = fraudScore + spamScore;
    
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

// CRITICAL FOR VERCEL: Export the app instead of calling app.listen()
export default app;
