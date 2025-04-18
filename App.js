require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const path = require('path');
const indexRouter = require('./routes/index');

const app = express();

// Middleware de base
app.use(cors({
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Accept']
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(helmet());
app.use(rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100
}));
app.use(express.json({ limit: '10kb' }));
app.use(express.urlencoded({ extended: true }));

// Logging simplifié
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
    next();
});

// Routes
app.use('/api', indexRouter);

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Gestion des erreurs
app.use((err, req, res, next) => {
    console.error(`[ERROR] ${err.stack}`);
    res.status(500).json({ 
        success: false,
        error: 'Erreur interne'
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`\nServeur démarré sur http://localhost:${PORT}`);
});