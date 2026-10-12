const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
  user_id: { type: Number, required: true, unique: true },
  name: { type: String, required: true },
  tagline: { type: String, default: '' },
  description: { type: String, default: '' },
  avatar: { type: String, default: '' },
  history_item_ids: [{ type: Number }]
}, { timestamps: true });

module.exports = mongoose.model('User', UserSchema);
