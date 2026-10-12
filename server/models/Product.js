const mongoose = require('mongoose');

const ProductSchema = new mongoose.Schema({
  id: { type: Number, required: true, unique: true },
  title: { type: String, required: true },
  category: { type: String, required: true },
  department: { type: String, default: 'Apparel' },
  price: { type: Number, required: true },
  rating: { type: Number, default: 4.5 },
  reviews_count: { type: Number, default: 0 },
  brand: { type: String, default: 'HYBRID Atelier' },
  description: { type: String, default: '' },
  image: { type: String, required: true },
  tags: [{ type: String }]
}, { timestamps: true });

module.exports = mongoose.model('Product', ProductSchema);
