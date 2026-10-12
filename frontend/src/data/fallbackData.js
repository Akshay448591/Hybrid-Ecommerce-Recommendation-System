/**
 * HYBRID Recommendation Engine - Embedded Fallback Dataset & Simulation Logic
 * Ensures immediate zero-latency rendering, cloud cold-start resilience,
 * and robust offline showcase behavior so 0 products is NEVER shown.
 */

export const FALLBACK_CATALOG = [
  {
    "id": 43493,
    "title": "Oversized Wool Trench Coat",
    "brand": "Aura Atelier",
    "category": "Outerwear & Coats",
    "department": "Outerwear",
    "price": 249.0,
    "rating": 4.3,
    "reviews_count": 24,
    "description": "Tailored double-breasted coat in rich brushed wool blend with wide lapels and storm flap.",
    "image": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "outerwear",
      "aura-atelier",
      "new-arrival"
    ]
  },
  {
    "id": 43494,
    "title": "Minimalist Quilted Puffer Jacket",
    "brand": "Kite & Thread",
    "category": "Outerwear & Coats",
    "department": "Outerwear",
    "price": 189.0,
    "rating": 4.4,
    "reviews_count": 41,
    "description": "Ultra-lightweight duck down insulation with matte water-repellent shell and fleece-lined pockets.",
    "image": "https://images.unsplash.com/photo-1544441893-675973e31985?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "outerwear",
      "kite-&-thread",
      "bestseller"
    ]
  },
  {
    "id": 1351,
    "title": "Classic Camel Overcoat",
    "brand": "Nordic Studio",
    "category": "Outerwear & Coats",
    "department": "Outerwear",
    "price": 289.0,
    "rating": 4.5,
    "reviews_count": 58,
    "description": "Timeless single-breasted silhouette cut from Italian wool cashmere blend with satin interior lining.",
    "image": "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "outerwear",
      "nordic-studio",
      "new-arrival"
    ]
  },
  {
    "id": 10078,
    "title": "Cropped Aviator Shearling Jacket",
    "brand": "Vanguard Denim",
    "category": "Outerwear & Coats",
    "department": "Outerwear",
    "price": 195.0,
    "rating": 4.6,
    "reviews_count": 75,
    "description": "Plush faux-shearling collar and buckle cuffs paired with structured vegan leather shell.",
    "image": "https://images.unsplash.com/photo-1551028719-00167b16eac5?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "outerwear",
      "vanguard-denim",
      "bestseller"
    ]
  },
  {
    "id": 17911,
    "title": "Technical Hooded Parka",
    "brand": "Solstice Minimal",
    "category": "Outerwear & Coats",
    "department": "Outerwear",
    "price": 219.0,
    "rating": 4.7,
    "reviews_count": 92,
    "description": "Seam-sealed stormproof shell featuring thermal insulation, two-way zipper, and adjustable storm visor.",
    "image": "https://images.unsplash.com/photo-1548883354-7622d03aca27?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "outerwear",
      "solstice-minimal",
      "new-arrival"
    ]
  },
  {
    "id": 33412,
    "title": "Brushed Herringbone Wool Blazer",
    "brand": "Arc & Loom",
    "category": "Outerwear & Coats",
    "department": "Outerwear",
    "price": 175.0,
    "rating": 4.8,
    "reviews_count": 109,
    "description": "Structured shoulder construction with notch lapel and horn buttons in timeless charcoal herringbone.",
    "image": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "outerwear",
      "arc-&-loom",
      "bestseller"
    ]
  },
  {
    "id": 1350,
    "title": "Pure Mongolian Cashmere Sweater",
    "brand": "Aura Atelier",
    "category": "Tops & Knitwear",
    "department": "Tops",
    "price": 165.0,
    "rating": 4.9,
    "reviews_count": 126,
    "description": "Featherlight 2-ply pure cashmere knit with ribbed crew neckline and relaxed drape.",
    "image": "https://images.unsplash.com/photo-1620799140408-edc6dcb6d633?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "tops",
      "aura-atelier",
      "new-arrival"
    ]
  },
  {
    "id": 10065,
    "title": "Relaxed Poplin Oxford Shirt",
    "brand": "Kite & Thread",
    "category": "Tops & Knitwear",
    "department": "Tops",
    "price": 85.0,
    "rating": 4.3,
    "reviews_count": 143,
    "description": "Pre-washed organic cotton poplin with crisp point collar and mother-of-pearl buttons.",
    "image": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "tops",
      "kite-&-thread",
      "bestseller"
    ]
  },
  {
    "id": 2434,
    "title": "Chunky Fisherman Ribbed Turtleneck",
    "brand": "Nordic Studio",
    "category": "Tops & Knitwear",
    "department": "Tops",
    "price": 145.0,
    "rating": 4.4,
    "reviews_count": 160,
    "description": "Heavyweight lambswool rib-knit sweater crafted with raglan sleeves and substantial rolled neck.",
    "image": "https://images.unsplash.com/photo-1576566588028-4147f3842f27?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "tops",
      "nordic-studio",
      "new-arrival"
    ]
  },
  {
    "id": 1774,
    "title": "Structured Heavyweight Tee",
    "brand": "Vanguard Denim",
    "category": "Tops & Knitwear",
    "department": "Tops",
    "price": 45.0,
    "rating": 4.5,
    "reviews_count": 177,
    "description": "Clean 280 GSM combed cotton crewneck t-shirt with boxy fit and reinforced binding.",
    "image": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "tops",
      "vanguard-denim",
      "bestseller"
    ]
  },
  {
    "id": 8,
    "title": "Merino Wool Polo Cardigan",
    "brand": "Solstice Minimal",
    "category": "Tops & Knitwear",
    "department": "Tops",
    "price": 125.0,
    "rating": 4.6,
    "reviews_count": 194,
    "description": "Fine-gauge extra-fine merino wool with mother-of-pearl button placket and ribbed hem.",
    "image": "https://images.unsplash.com/photo-1583743814966-8936f5b7be1a?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "tops",
      "solstice-minimal",
      "new-arrival"
    ]
  },
  {
    "id": 34697,
    "title": "Silk Blend Relaxed Button-Down",
    "brand": "Arc & Loom",
    "category": "Tops & Knitwear",
    "department": "Tops",
    "price": 115.0,
    "rating": 4.7,
    "reviews_count": 211,
    "description": "Fluid silk-viscose blend drape with camp collar and subtle sand-washed sheen.",
    "image": "https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "tops",
      "arc-&-loom",
      "bestseller"
    ]
  },
  {
    "id": 11382,
    "title": "Selvedge Straight Leg Denim",
    "brand": "Aura Atelier",
    "category": "Bottoms & Denim",
    "department": "Bottoms",
    "price": 135.0,
    "rating": 4.8,
    "reviews_count": 228,
    "description": "Authentic 13.5oz Japanese selvedge denim in raw indigo wash with clean copper rivets.",
    "image": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "bottoms",
      "aura-atelier",
      "new-arrival"
    ]
  },
  {
    "id": 17910,
    "title": "Pleated Wool Flannel Trousers",
    "brand": "Kite & Thread",
    "category": "Bottoms & Denim",
    "department": "Bottoms",
    "price": 155.0,
    "rating": 4.9,
    "reviews_count": 245,
    "description": "Double reverse pleats with straight leg profile in Italian wool blend with coin pocket.",
    "image": "https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "bottoms",
      "kite-&-thread",
      "bestseller"
    ]
  },
  {
    "id": 49642,
    "title": "Relaxed Linen Drawstring Pants",
    "brand": "Nordic Studio",
    "category": "Bottoms & Denim",
    "department": "Bottoms",
    "price": 95.0,
    "rating": 4.3,
    "reviews_count": 262,
    "description": "Breathable 100% Belgian flax linen with elasticated drawstring waistband for effortless ease.",
    "image": "https://images.unsplash.com/photo-1506629082955-511b1aa562c8?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "bottoms",
      "nordic-studio",
      "new-arrival"
    ]
  },
  {
    "id": 50215,
    "title": "High-Rise Wide Leg Jeans",
    "brand": "Vanguard Denim",
    "category": "Bottoms & Denim",
    "department": "Bottoms",
    "price": 110.0,
    "rating": 4.4,
    "reviews_count": 279,
    "description": "Vintage stonewash denim featuring flattering high rise, relaxed thigh, and clean hem.",
    "image": "https://images.unsplash.com/photo-1582552938357-32b906df40cb?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "bottoms",
      "vanguard-denim",
      "bestseller"
    ]
  },
  {
    "id": 65881,
    "title": "Structured Cotton Chinos",
    "brand": "Solstice Minimal",
    "category": "Bottoms & Denim",
    "department": "Bottoms",
    "price": 88.0,
    "rating": 4.5,
    "reviews_count": 296,
    "description": "Mid-weight stretch cotton twill with tapered leg and stain-resistant treatment.",
    "image": "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "bottoms",
      "solstice-minimal",
      "new-arrival"
    ]
  },
  {
    "id": 66201,
    "title": "Tailored Wool Blend Shorts",
    "brand": "Arc & Loom",
    "category": "Bottoms & Denim",
    "department": "Bottoms",
    "price": 85.0,
    "rating": 4.6,
    "reviews_count": 313,
    "description": "Sophisticated knee-length shorts in crisp tropical wool with belt loops and slash pockets.",
    "image": "https://images.unsplash.com/photo-1591195853828-11db59a44f6b?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "bottoms",
      "arc-&-loom",
      "bestseller"
    ]
  },
  {
    "id": 43495,
    "title": "Bias-Cut Silk Slip Dress",
    "brand": "Aura Atelier",
    "category": "Dresses & Jumpsuits",
    "department": "Dresses",
    "price": 185.0,
    "rating": 4.7,
    "reviews_count": 330,
    "description": "Gracefully draped 100% mulberry silk with delicate spaghetti straps and fluid midi hem.",
    "image": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "dresses",
      "aura-atelier",
      "new-arrival"
    ]
  },
  {
    "id": 13260,
    "title": "Pleated Floral Chiffon Midi Dress",
    "brand": "Kite & Thread",
    "category": "Dresses & Jumpsuits",
    "department": "Dresses",
    "price": 160.0,
    "rating": 4.8,
    "reviews_count": 347,
    "description": "Cascading accordion pleats with subtle botanical watercolor print and blouson sleeves.",
    "image": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "dresses",
      "kite-&-thread",
      "bestseller"
    ]
  },
  {
    "id": 63777,
    "title": "Sculpted Ribbed Knit Maxi Dress",
    "brand": "Nordic Studio",
    "category": "Dresses & Jumpsuits",
    "department": "Dresses",
    "price": 130.0,
    "rating": 4.9,
    "reviews_count": 364,
    "description": "Body-contouring compact rib knit with elegant mock turtleneck and walking slit.",
    "image": "https://images.unsplash.com/photo-1568252542512-9fe8fe9c87bb?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "dresses",
      "nordic-studio",
      "new-arrival"
    ]
  },
  {
    "id": 13419,
    "title": "Tailored Belted Utility Jumpsuit",
    "brand": "Vanguard Denim",
    "category": "Dresses & Jumpsuits",
    "department": "Dresses",
    "price": 168.0,
    "rating": 4.3,
    "reviews_count": 31,
    "description": "Crisp cotton twill jumpsuit featuring D-ring waist belt, flap pockets, and notched collar.",
    "image": "https://images.unsplash.com/photo-1502716119720-b23a93e5fe1b?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "dresses",
      "vanguard-denim",
      "bestseller"
    ]
  },
  {
    "id": 60751,
    "title": "Linen Tiered Bohemian Dress",
    "brand": "Solstice Minimal",
    "category": "Dresses & Jumpsuits",
    "department": "Dresses",
    "price": 140.0,
    "rating": 4.4,
    "reviews_count": 48,
    "description": "Airy airy garment-washed linen with ruffled tiers, keyhole neckline, and side pockets.",
    "image": "https://images.unsplash.com/photo-1496747611176-843222e1e57c?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "dresses",
      "solstice-minimal",
      "new-arrival"
    ]
  },
  {
    "id": 6845,
    "title": "Asymmetrical Evening Gown",
    "brand": "Arc & Loom",
    "category": "Dresses & Jumpsuits",
    "department": "Dresses",
    "price": 265.0,
    "rating": 4.5,
    "reviews_count": 65,
    "description": "Floor-sweeping satin crepe with one-shoulder neckline and thigh-high side slit.",
    "image": "https://images.unsplash.com/photo-1518895949257-7621c3c786d7?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "dresses",
      "arc-&-loom",
      "bestseller"
    ]
  },
  {
    "id": 9,
    "title": "Minimalist Calfskin Sneaker",
    "brand": "Aura Atelier",
    "category": "Footwear & Shoes",
    "department": "Footwear",
    "price": 145.0,
    "rating": 4.6,
    "reviews_count": 82,
    "description": "Italian nappa leather upper with vulcanized rubber sole and cushioned memory foam insole.",
    "image": "https://images.unsplash.com/photo-1560769629-975ec94e6a86?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "footwear",
      "aura-atelier",
      "new-arrival"
    ]
  },
  {
    "id": 61656,
    "title": "Chelsea Boot in Suede",
    "brand": "Kite & Thread",
    "category": "Footwear & Shoes",
    "department": "Footwear",
    "price": 195.0,
    "rating": 4.7,
    "reviews_count": 99,
    "description": "Handcrafted Portuguese suede with Goodyear welt, elastic side gussets, and pull tab.",
    "image": "https://images.unsplash.com/photo-1638247025967-b4e38f787b76?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "footwear",
      "kite-&-thread",
      "bestseller"
    ]
  },
  {
    "id": 60129,
    "title": "Lug-Sole Leather Loafers",
    "brand": "Nordic Studio",
    "category": "Footwear & Shoes",
    "department": "Footwear",
    "price": 175.0,
    "rating": 4.8,
    "reviews_count": 116,
    "description": "Polished spazzolato leather with chunky commando lug sole and penny keeper slot.",
    "image": "https://images.unsplash.com/photo-1614252235316-8c857d38b5f4?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "footwear",
      "nordic-studio",
      "new-arrival"
    ]
  },
  {
    "id": 10718,
    "title": "Strappy Heeled Sandal",
    "brand": "Vanguard Denim",
    "category": "Footwear & Shoes",
    "department": "Footwear",
    "price": 135.0,
    "rating": 4.9,
    "reviews_count": 133,
    "description": "Graceful barely-there nappa straps with 70mm block heel and squared open toe.",
    "image": "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "footwear",
      "vanguard-denim",
      "bestseller"
    ]
  },
  {
    "id": 17492,
    "title": "Performance Knit Running Shoe",
    "brand": "Solstice Minimal",
    "category": "Footwear & Shoes",
    "department": "Footwear",
    "price": 160.0,
    "rating": 4.3,
    "reviews_count": 150,
    "description": "Engineered breathable mesh upper with responsive carbon-infused foam midsole.",
    "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "footwear",
      "solstice-minimal",
      "new-arrival"
    ]
  },
  {
    "id": 63822,
    "title": "Leather Combat Boots",
    "brand": "Arc & Loom",
    "category": "Footwear & Shoes",
    "department": "Footwear",
    "price": 215.0,
    "rating": 4.4,
    "reviews_count": 167,
    "description": "Durable full-grain leather with waxed laces, inside zipper, and treaded vibram sole.",
    "image": "https://images.unsplash.com/photo-1520639888713-7851133b1ed0?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "footwear",
      "arc-&-loom",
      "bestseller"
    ]
  },
  {
    "id": 1768,
    "title": "Structured Leather Crossbody Bag",
    "brand": "Aura Atelier",
    "category": "Accessories & Bags",
    "department": "Accessories",
    "price": 175.0,
    "rating": 4.5,
    "reviews_count": 184,
    "description": "Full-grain Italian calf leather with accordion interior, magnetic clasp, and gold hardware.",
    "image": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "accessories",
      "aura-atelier",
      "new-arrival"
    ]
  },
  {
    "id": 35753,
    "title": "Canvas & Leather Weekend Duffle",
    "brand": "Kite & Thread",
    "category": "Accessories & Bags",
    "department": "Accessories",
    "price": 195.0,
    "rating": 4.6,
    "reviews_count": 201,
    "description": "Heavy duty 24oz water-resistant duck canvas with saddle leather straps and brass hardware.",
    "image": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "accessories",
      "kite-&-thread",
      "bestseller"
    ]
  },
  {
    "id": 59,
    "title": "Cashmere Wool Fringe Scarf",
    "brand": "Nordic Studio",
    "category": "Accessories & Bags",
    "department": "Accessories",
    "price": 85.0,
    "rating": 4.7,
    "reviews_count": 218,
    "description": "Generously proportioned 100% cashmere scarf in heather grey with twisted fringe ends.",
    "image": "https://images.unsplash.com/photo-1608256246200-53e635b5b65f?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "accessories",
      "nordic-studio",
      "new-arrival"
    ]
  },
  {
    "id": 63979,
    "title": "Handcrafted Acetate Sunglasses",
    "brand": "Vanguard Denim",
    "category": "Accessories & Bags",
    "department": "Accessories",
    "price": 125.0,
    "rating": 4.8,
    "reviews_count": 235,
    "description": "Classic square silhouette cut from Italian Mazzucchelli acetate with polarized UV400 lenses.",
    "image": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "accessories",
      "vanguard-denim",
      "bestseller"
    ]
  },
  {
    "id": 844,
    "title": "Woven Leather Tote Bag",
    "brand": "Solstice Minimal",
    "category": "Accessories & Bags",
    "department": "Accessories",
    "price": 220.0,
    "rating": 4.9,
    "reviews_count": 252,
    "description": "Intricately hand-woven nappa leather basket tote with detachable zip pouch.",
    "image": "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "accessories",
      "solstice-minimal",
      "new-arrival"
    ]
  },
  {
    "id": 64780,
    "title": "Minimalist Automatic Watch",
    "brand": "Arc & Loom",
    "category": "Accessories & Bags",
    "department": "Accessories",
    "price": 240.0,
    "rating": 4.3,
    "reviews_count": 269,
    "description": "Brushed 316L stainless steel case with sapphire crystal, clean Bauhaus dial, and Horween strap.",
    "image": "https://images.unsplash.com/photo-1524805444758-089113d48a6d?auto=format&fit=crop&w=800&q=80",
    "in_stock": true,
    "tags": [
      "accessories",
      "arc-&-loom",
      "bestseller"
    ]
  }
];

export const FALLBACK_PERSONAS = [
  {
    "user_id": 0,
    "name": "Sophia Martinez",
    "tagline": "Minimalist Chic & Elevated Essentials",
    "description": "Prefers clean silhouettes, neutral palettes, fine cashmere knitwear, and tailored wool overcoats.",
    "avatar": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=300&q=80",
    "history_item_ids": [
      43493,
      1350,
      11382
    ],
    "preferred_category": "Tops & Knitwear"
  },
  {
    "user_id": 1,
    "name": "Marcus Vance",
    "tagline": "Contemporary Streetwear & Heavy Denim",
    "description": "Loves relaxed fit raw selvedge denim, heavy cotton t-shirts, aviator jackets, and lug-sole loafers.",
    "avatar": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=300&q=80",
    "history_item_ids": [
      43494,
      17910,
      61656
    ],
    "preferred_category": "Bottoms & Denim"
  },
  {
    "user_id": 6,
    "name": "Elena Rostova",
    "tagline": "Luxury Eveningwear & Structured Elegance",
    "description": "Focuses on premium silk dresses, tailored blazers, leather bags, and statement evening pieces.",
    "avatar": "https://images.unsplash.com/photo-1517841905240-472988babdf9?auto=format&fit=crop&w=300&q=80",
    "history_item_ids": [
      1351,
      43495,
      1768
    ],
    "preferred_category": "Dresses & Jumpsuits"
  },
  {
    "user_id": 8,
    "name": "David Kim",
    "tagline": "Athletic Motion & Technical Outerwear",
    "description": "Values breathable technical parkas, performance knit footwear, and modern utility wear.",
    "avatar": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=300&q=80",
    "history_item_ids": [
      17911,
      17492,
      65881
    ],
    "preferred_category": "Outerwear & Coats"
  },
  {
    "user_id": 16,
    "name": "Chloe Bennett",
    "tagline": "Bohemian Artistry & Crafted Accessories",
    "description": "Appreciates hand-woven leather totes, fringe cashmere scarves, linen pieces, and acetate eyewear.",
    "avatar": "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?auto=format&fit=crop&w=300&q=80",
    "history_item_ids": [
      60751,
      59,
      844
    ],
    "preferred_category": "Accessories & Bags"
  },
  {
    "user_id": -1,
    "name": "Guest Explorer",
    "tagline": "Cold-Start New Visitor (Session-Driven)",
    "description": "First-time visitor without interaction graph history; recommendations adapt live as items are clicked or added to cart via SASRec.",
    "avatar": "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=300&q=80",
    "history_item_ids": [],
    "preferred_category": "All"
  }
];

export function getFallbackProducts({
  category = 'all',
  sortBy = 'popular',
  limit = 50,
  offset = 0,
  query = ''
} = {}) {
  let results = [...FALLBACK_CATALOG];

  if (category && category.toLowerCase() !== 'all') {
    results = results.filter(
      item => item.category && item.category.toLowerCase() === category.toLowerCase()
    );
  }

  if (query && query.trim()) {
    const q = query.toLowerCase().trim();
    results = results.filter(item =>
      (item.title && item.title.toLowerCase().includes(q)) ||
      (item.brand && item.brand.toLowerCase().includes(q)) ||
      (item.description && item.description.toLowerCase().includes(q)) ||
      (item.tags && item.tags.some(t => t.toLowerCase().includes(q)))
    );
  }

  if (sortBy === 'price_asc') {
    results.sort((a, b) => a.price - b.price);
  } else if (sortBy === 'price_desc') {
    results.sort((a, b) => b.price - a.price);
  } else if (sortBy === 'rating') {
    results.sort((a, b) => b.rating - a.rating);
  } else if (sortBy === 'popular') {
    results.sort((a, b) => (b.reviews_count || 0) - (a.reviews_count || 0));
  }

  const total = results.length;
  const paged = results.slice(offset, offset + limit);

  return {
    total,
    items: paged,
    limit,
    offset
  };
}

export function generateFallbackRecommendations({
  userId = 0,
  sessionSequence = [],
  topK = 12,
  mode = 'hybrid_v2',
  graphWeight = 1.0,
  seqWeight = 1.0
} = {}) {
  const persona = FALLBACK_PERSONAS.find(p => p.user_id === userId) || FALLBACK_PERSONAS[0];
  const preferredCat = persona?.preferred_category || 'Tops & Knitwear';
  const historyIds = new Set(persona?.history_item_ids || []);
  const sessionIds = new Set(sessionSequence || []);

  const scoredItems = FALLBACK_CATALOG.map(item => {
    // Graph Collaborative affinity (LightGCN simulation)
    let gScore = 0.15;
    if (historyIds.has(item.id)) gScore += 0.45;
    if (item.category === preferredCat) gScore += 0.35;
    if (item.rating > 4.5) gScore += 0.1;
    gScore += ((item.id % 17) / 100);

    // Sequential Self-Attention affinity (SASRec simulation)
    let sScore = 0.1;
    if (sessionSequence.length > 0) {
      const lastSessionItem = FALLBACK_CATALOG.find(x => x.id === sessionSequence[sessionSequence.length - 1]);
      if (lastSessionItem) {
        if (lastSessionItem.category === item.category) sScore += 0.55;
        if (lastSessionItem.brand === item.brand) sScore += 0.25;
      }
      if (sessionIds.has(item.id)) sScore += 0.2;
    } else {
      sScore = gScore * 0.7;
    }
    sScore += ((item.id % 23) / 100);

    // Mode handling
    let finalGraph = gScore;
    let finalSeq = sScore;
    if (mode === 'pure_lightgcn') {
      finalSeq = 0.0;
    } else if (mode === 'pure_sasrec') {
      finalGraph = 0.0;
    }

    const weightedGraph = finalGraph * graphWeight;
    const weightedSeq = finalSeq * seqWeight;
    const crossInteraction = (weightedGraph * weightedSeq) * 0.4;
    const totalScore = weightedGraph + weightedSeq + crossInteraction;

    const gRaw = Math.max(0.01, weightedGraph);
    const sRaw = Math.max(0.01, weightedSeq);
    const cRaw = Math.max(0.005, crossInteraction);
    const sumRaw = gRaw + sRaw + cRaw;

    const gPct = Math.round((gRaw / sumRaw) * 100);
    const sPct = Math.round((sRaw / sumRaw) * 100);
    const cPct = Math.max(0, 100 - gPct - sPct);

    let explanation = '';
    if (sPct > 55) {
      explanation = 'High SASRec sequential attention triggered by your active browsing session interactions.';
    } else if (gPct > 55) {
      explanation = 'Strong LightGCN collaborative graph affinity matching historical co-interaction manifolds.';
    } else {
      explanation = 'Balanced Hybrid V2 fusion integrating long-term collaborative profile and real-time session transitions.';
    }

    return {
      ...item,
      _totalScore: totalScore,
      recommendation_metadata: {
        rank: 1,
        hybrid_score: parseFloat(totalScore.toFixed(4)),
        graph_score: parseFloat(gScore.toFixed(4)),
        sequential_score: parseFloat(sScore.toFixed(4)),
        cross_domain_score: parseFloat(crossInteraction.toFixed(4)),
        contributions: {
          graph_percent: gPct,
          sequential_percent: sPct,
          cross_percent: cPct
        },
        explanation
      }
    };
  });

  // Sort descending by score
  scoredItems.sort((a, b) => b._totalScore - a._totalScore);

  // Exclude immediate duplicate in history/session unless catalog is small
  const filtered = scoredItems.slice(0, topK).map((item, index) => ({
    ...item,
    recommendation_metadata: {
      ...item.recommendation_metadata,
      rank: index + 1
    }
  }));

  return filtered;
}
