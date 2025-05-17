# Keyword Research Tool

This tool helps with deep keyword research by analyzing SERPs, competitor content, and generating insights using Google's Gemini AI.

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
4. Run the application:
   ```
   python src/app.py
   ```
5. Access the web interface at `http://localhost:5000`

## Key Features

- SERP analysis with competitor content parsing
- Keyword intent classification
- Content opportunity identification
- SERP feature analysis
- Automated research insights
- Content strategy recommendations

## API Usage

Send a POST request to `/api/research` with a JSON body containing:
```json
{
  "seed_keyword": "your main keyword",
  "secondary_keywords": ["keyword1", "keyword2"],
  "industry": "your industry"
}
```

## Dependencies

This tool uses:
- Google Gemini AI for language processing
- Browser-use for web scraping
- Flask for the web interface

## Environment Variables

- `GOOGLE_API_KEY`: Your Google API key for accessing Gemini models
