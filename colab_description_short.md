# 🏠 Photo Metadata AI - AWS Rekognition Photo Categorizer

## 📋 What It Does

Automatically analyzes and categorizes real estate photos using AWS Rekognition AI. Uploads images from local folders to S3, then uses AI to detect objects and scenes, organizing them into categories like kitchen, bedroom, bathroom, etc.

## 🎯 Categories

- 🍳 **Kitchen**: Appliances, cabinets, countertops
- 🛏️ **Bedroom**: Beds, furniture, sleeping areas  
- 🚿 **Bathroom**: Toilets, showers, sinks, mirrors
- 🛋️ **Living Room**: Sofas, TVs, fireplaces
- 🍽️ **Dining Room**: Dining tables, chairs
- 🏠 **Exterior**: Building exteriors, architecture
- 🚗 **Garage**: Cars, vehicles, parking
- 💼 **Office**: Desks, computers, work areas
- 👕 **Laundry**: Washing machines, dryers
- 🪜 **Stairs**: Staircases, railings
- 👔 **Closet**: Wardrobes, clothing storage
- 🏊 **Pool**: Swimming pools, water features
- 🌿 **Balcony**: Terraces, patios, decks
- 📦 **Other**: Unmatched items

## 📁 Required Folder Structure

```
images/
├── property-123/
│   ├── kitchen1.jpg
│   ├── bedroom1.jpg
│   └── bathroom1.jpg
├── property-456/
│   ├── exterior1.jpg
│   └── garage1.jpg
└── property-789/
    ├── office1.jpg
    └── dining1.jpg
```

## 🔧 Usage Options

When you run `photo-categorizer`, you'll get three options:

1. **📤 Upload Only**: Upload images from local folder to S3
2. **🔍 Categorize Only**: Process existing images in S3  
3. **🚀 Upload + Categorize**: Complete workflow (recommended)

## 📊 Results

- ✅ **Organized Images**: Sorted into category folders in S3
- 📈 **JSON Reports**: Detailed analysis with confidence scores
- 📋 **Summary**: Breakdown of images by category
- 🔍 **Labels**: Top detected objects for each image

## 🛠️ Requirements

- ✅ AWS Account with S3 and Rekognition access
- ✅ AWS credentials configured
- ✅ S3 bucket created
- ✅ Images in proper folder structure

## 🔐 Security Notes

- ⚠️ Never commit AWS credentials to version control
- 🔒 Use IAM roles with minimal required permissions 