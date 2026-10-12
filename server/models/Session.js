const mongoose = require('mongoose');

const SessionSchema = new mongoose.Schema({
  sessionId: { type: String, required: true, unique: true },
  userId: { type: Number, default: 0 },
  interactionSequence: [{
    itemId: { type: Number, required: true },
    timestamp: { type: Date, default: Date.now }
  }],
  cart: [{
    productId: { type: Number, required: true },
    addedAt: { type: Date, default: Date.now }
  }]
}, { timestamps: true });

module.exports = mongoose.model('Session', SessionSchema);
