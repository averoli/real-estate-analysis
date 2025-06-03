# Real Estate Analysis App

This Streamlit application helps analyze real estate data by importing from various sources including Excel files, CSV files, and Google Sheets.

## Features

- Import data from multiple Excel/CSV files
- Import data from Google Sheets
- Automatic header detection
- Smart column mapping
- Price analysis by district
- Identification of good deals (properties below market average)
- Export analysis results to Excel

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For Google Sheets integration:

   a. Go to [Google Cloud Console](https://console.cloud.google.com/)
   b. Create a new project
   c. Enable Google Sheets API for your project
   d. Create a Service Account:
      - Go to "IAM & Admin" > "Service Accounts"
      - Click "Create Service Account"
      - Fill in the details and create
   e. Create a key for the Service Account:
      - Select your service account
      - Go to "Keys" tab
      - Add Key > Create new key > JSON
      - Download the key file
   f. Rename the downloaded JSON file to `google_sheets_credentials.json`
   g. Place it in the root directory of the application
   h. Share your Google Sheet with the service account email address (found in the JSON file)

3. Run the application:
```bash
streamlit run multi_import_app.py
```

## Using Google Sheets

1. Create or open your Google Sheet
2. Share it with the service account email address
3. Copy the Google Sheets URL
4. Paste the URL in the application's URL input field

## File Format Requirements

Your data should include these columns (in any language):
- Project/Complex name
- District/Area
- Size (sqm)
- Number of bedrooms
- Number of floors
- Price

The app will try to automatically detect and map these columns from your data. 