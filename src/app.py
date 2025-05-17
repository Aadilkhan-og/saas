# Filename: src/app.py
import asyncio
import os
import logging
import sys
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Configure logging with proper encoding handling
def setup_logging():
    """Configure logging with proper encoding handling for Windows console"""
    log_filename = f"keyword_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Configure root logger
    logger = logging.getLogger()
    # Setting level to DEBUG during development for more verbose output
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to avoid duplication
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # File handler (supports unicode)
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # Log DEBUG to file
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler with encoding error handling
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Keep console less verbose, INFO level
    # Custom formatter to handle emoji characters in logs (copied from original)
    class EmojiFilter(logging.Formatter):
        """Custom formatter to handle emoji characters in logs"""
        def format(self, record):
            msg = super().format(record)
            # Replace emoji characters with text equivalents
            emoji_replacements = {
                '\U0001f680': '[ROCKET]',  # 🚀
                '\U0001f4cd': '[PIN]',     # 📍
                '\U00002705': '[CHECK]',   # ✅
                '\U0001f4dd': '[MEMO]',    # 📝
                '\U0001f50d': '[SEARCH]',  # 🔍
                '\U0001f6a8': '[ALERT]',   # 🚨
                '\U00002139': '[INFO]',    # ℹ️
                '\U000026a0': '[WARNING]', # ⚠️
                '\U0000274c': '[ERROR]',   # ❌
                '\U0001f44d': '[THUMBSUP]',# 👍
                '\U0001F937': '[SHRUG]',   # 🤷
                '\u26A0\uFE0F': '[WARNING]', # ⚠️ (Handling variation)
                '\U0001F517': '[LINK]', # 🔗
                '\U0001F5B1\uFE0F': '[MOUSE]', # 🖱️
                '\U0001F336\uFE0F': '[HOTPEPPER]', # 🌶️
                '\u2764\uFE0F': '[HEART]', # ❤️
            }

            for emoji, replacement in emoji_replacements.items():
                msg = msg.replace(emoji, replacement)
            return msg

    console_formatter = EmojiFilter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Original format had emojis directly, using the EmojiFilter
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Also configure browser-use logger specifically
    browser_logger = logging.getLogger("browser_use")
    browser_logger.propagate = False  # Don't propagate to root logger to avoid duplication
    browser_logger.addHandler(file_handler)
    browser_logger.addHandler(console_handler)

    return logger

# Set up logging
logger = setup_logging()
app_logger = logging.getLogger("keyword_research")

# Import modules
from modules.input_handler import InputHandler
from modules.serp_collector import SerpCollector
from modules.content_analyzer import ContentAnalyzer
from modules.keyword_processor import KeywordProcessor
from modules.insight_generator import InsightGenerator
from modules.result_renderer import ResultRenderer

try:
    from dotenv import load_dotenv
    load_dotenv()
    app_logger.info("Loaded environment variables from .env file")
except ImportError:
    app_logger.warning("python-dotenv not installed. Using existing environment variables.")
except Exception as e:
    app_logger.error(f"Error loading .env file: {str(e)}")

# Import GeminiClient here as well to check availability
try:
    from modules.gemini_client import GeminiClient
    GEMINI_AVAILABLE = GeminiClient.is_available()
except ImportError:
    GEMINI_AVAILABLE = False
# Import Agent here as well to check availability
try:
    from browser_use import Agent
except ImportError:
    Agent = None


app = Flask(__name__)
CORS(app)

# Initialize components
input_handler = None
serp_collector = None
content_analyzer = None
keyword_processor = None
insight_generator = None
result_renderer = None

def init_components():
    """Initialize all components with proper error handling"""
    global input_handler, serp_collector, content_analyzer, keyword_processor, insight_generator, result_renderer

    try:
        # Check for Google API key
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            app_logger.warning("GOOGLE_API_KEY not found in environment. LLM features may not work correctly.")

        # Check if core dependencies are met
        if not GEMINI_AVAILABLE or not Agent:
            app_logger.error("Core dependencies (google-generativeai, langchain-google-genai, browser_use) are not installed. Cannot initialize components.")
            return False

        # Initialize components
        input_handler = InputHandler()
        app_logger.info("InputHandler initialized")

        serp_collector = SerpCollector()
        app_logger.info("SerpCollector initialized")

        content_analyzer = ContentAnalyzer()
        app_logger.info("ContentAnalyzer initialized")

        keyword_processor = KeywordProcessor()
        app_logger.info("KeywordProcessor initialized")

        insight_generator = InsightGenerator()
        app_logger.info("InsightGenerator initialized")

        result_renderer = ResultRenderer()
        app_logger.info("ResultRenderer initialized")

        return True
    except Exception as e:
        app_logger.error(f"Error initializing components: {str(e)}")
        return False

# Attempt to initialize components on startup
# If it fails, the research endpoint will return an error
init_success = init_components()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/research', methods=['POST'])
async def research():
    """
    Main endpoint for the deep keyword research feature
    """
    start_time = datetime.now()
    app_logger.info(f"Received research request at {start_time}")

    # Check if components initialized successfully on startup
    if not init_success:
        app_logger.error("Research request received but components failed to initialize on startup.")
        return jsonify({"error": "Server components not initialized. Check server logs for details."}), 500

    # Check if key components required for this route are available
    if not all([input_handler, serp_collector, content_analyzer, keyword_processor, insight_generator, result_renderer]):
         app_logger.error("Research request received but necessary components are missing or failed to initialize.")
         return jsonify({"error": "Server components missing. Cannot perform research."}), 500

    try:
        # Get request data
        data = request.get_json()
        app_logger.info(f"Received request data: {data}")

        # Process input
        processed_input = input_handler.process_input(data)
        app_logger.info(f"Processed input data. Seed keyword: {processed_input['seed_keyword']}, Total keywords: {len(processed_input['keywords'])}")

        # Execute workflow with timeouts and retries
        # 1. Collect SERP data
        serp_data = {} # Initialize with empty dict
        try:
            # Use a longer timeout for SERP collection as it involves browser actions
            serp_data = await asyncio.wait_for(
                serp_collector.collect_serp_data(processed_input['keywords']),
                timeout=300  # 5-minute timeout for SERP collection
            )
            app_logger.info(f"SERP data collected. Keywords processed: {len(serp_data.get('serp_data', {}))}")
        except asyncio.TimeoutError:
            app_logger.error("SERP data collection timed out.")
            # Continue with partial/empty serp_data
            serp_data = {
                "keywords": processed_input['keywords'],
                "serp_data": {}, # This holds processed results lists
                "features": {}, # Aggregated features
                "paa_questions": {}, # Aggregated PAA
                "related_searches": {}, # Aggregated related searches
                "top_urls": [], # Collected URLs
                 # Add raw data structures, empty if timeout occurred
                "raw_serp_results": None,
                "raw_paa_questions": None,
                "raw_related_searches": None,
                "raw_features": None,
            }
            # You might want to add a status flag to the final results indicating timeout

        except Exception as e:
            app_logger.error(f"Error collecting SERP data: {str(e)}", exc_info=True)
            # Continue with empty data rather than failing completely
            serp_data = {
                "keywords": processed_input['keywords'],
                "serp_data": {},
                "features": {},
                "paa_questions": {},
                "related_searches": {},
                "top_urls": [],
                 # Add raw data structures, empty if error occurred
                "raw_serp_results": None,
                "raw_paa_questions": None,
                "raw_related_searches": None,
                "raw_features": None,
            }


        # 2. Analyze competitor content
        competitor_data = {} # Initialize with empty dict
        # Only attempt content analysis if we got some top URLs from SERP collection
        urls_to_analyze = serp_data.get('top_urls', [])
        if urls_to_analyze:
            try:
                 # Use a longer timeout for content analysis per URL
                competitor_data = await asyncio.wait_for(
                    content_analyzer.analyze_content(urls_to_analyze),
                    timeout=300 + (len(urls_to_analyze) * 60) # Base 5 min + 1 min per URL
                )
                app_logger.info(f"Competitor content analyzed. URLs analyzed: {len(competitor_data.get('analyzed_urls', []))}")
            except asyncio.TimeoutError:
                app_logger.error("Content analysis timed out.")
                # Continue with empty data
                competitor_data = {
                    "analyzed_urls": [],
                    "content_analysis": {}, # Detailed analysis per URL
                    "common_themes": [], # Aggregated themes
                    "content_types": {}, # Aggregated types count
                    "heading_structure": {}, # Detailed headings (per URL)
                    "content_length": {}, # Detailed word count (per URL)
                    "summary": {}, # Aggregated summary
                }
                # You might want to add a status flag to the final results indicating timeout
            except Exception as e:
                app_logger.error(f"Error analyzing competitor content: {str(e)}", exc_info=True)
                # Continue with empty data
                competitor_data = {
                    "analyzed_urls": [],
                    "content_analysis": {}, # Detailed analysis per URL
                    "common_themes": [], # Aggregated themes
                    "content_types": {}, # Aggregated types count
                    "heading_structure": {}, # Detailed headings (per URL)
                    "content_length": {}, # Detailed word count (per URL)
                    "summary": {}, # Aggregated summary
                }
        else:
            app_logger.warning("No top URLs collected from SERP, skipping content analysis.")
            competitor_data = {
                "analyzed_urls": [],
                "content_analysis": {},
                "common_themes": [],
                "content_types": {},
                "heading_structure": {},
                "content_length": {},
                "summary": {},
            }


        # 3. Process keywords (classification and clustering)
        keyword_analysis = {} # Initialize with empty dict
        try:
            keyword_analysis = keyword_processor.process_keywords(
                processed_input['keywords'],
                serp_data, # Pass potentially incomplete serp_data
                competitor_data # Pass potentially incomplete competitor_data
            )
            app_logger.info(f"Keywords processed. Intents classified: {len(keyword_analysis.get('intent_classification', {}))}, Clusters: {len(keyword_analysis.get('clusters', []))}")
        except Exception as e:
            app_logger.error(f"Error processing keywords: {str(e)}", exc_info=True)
            # Continue with default data
            keyword_analysis = {
                "intent_classification": {},
                "clusters": [],
                "keyword_scores": {},
                "question_keywords": [],
            }


        # 4. Generate insights
        insights = {} # Initialize with empty dict
        try:
            insights = insight_generator.generate_insights(
                keyword_analysis,
                serp_data, # Pass potentially incomplete serp_data
                competitor_data # Pass potentially incomplete competitor_data
            )
            app_logger.info("Insights generated successfully")
        except Exception as e:
            app_logger.error(f"Error generating insights: {str(e)}", exc_info=True)
            # Continue with default insights
            insights = {
                "content_opportunities": [],
                "serp_feature_insights": [],
                "competitive_landscape": {},
                "keyword_recommendations": [],
                "topic_clusters": [],
                "intent_distribution": {},
                "summary": "Unable to generate complete insights due to an error."
            }


        # 5. Render results (Prepare the final JSON structure)
        results = {} # Initialize with empty dict
        try:
             # Pass ALL collected/processed data to the renderer
            results = result_renderer.render_results(
                processed_input,
                keyword_analysis,
                serp_data, # Aggregated SERP data for insight context
                competitor_data, # Aggregated competitor data for insight context
                insights,
                serp_data, # Pass full serp_data for detailed view
                competitor_data # Pass full competitor_data for detailed view
            )
            app_logger.info("Results structure prepared successfully")
        except Exception as e:
            app_logger.error(f"Error rendering results structure: {str(e)}", exc_info=True)
            # Return a simplified response in case of error
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            return jsonify({
                "error": "Error preparing final results structure",
                "details": str(e),
                "summary": {
                    "title": f"Keyword Research: {processed_input.get('seed_keyword', 'Unknown')}",
                    "date": end_time.strftime("%Y-%m-%d"),
                    "keyword_count": len(processed_input.get('keywords', [])),
                    "processing_time": f"{processing_time:.2f} seconds",
                    "insight_summary": "An error occurred during the research process.",
                    "top_opportunities": []
                },
                 # Include any data collected up to this point for debugging
                "partial_serp_data": serp_data,
                "partial_competitor_data": competitor_data,
                "partial_keyword_analysis": keyword_analysis,
                "partial_insights": insights,

            }), 500


        # Calculate and log processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        app_logger.info(f"Research request completed in {processing_time:.2f} seconds")

        # Add processing metadata to results (Ensure metadata exists)
        if 'metadata' not in results:
             results['metadata'] = {}

        results["metadata"].update({
            "processing_time_seconds": processing_time,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            # Use counts from the actual data collected
            "serp_keywords_processed": len(serp_data.get('serp_data', {})),
            "urls_analyzed": len(competitor_data.get('analyzed_urls', [])),
            "version": "1.0.2" # Keep version consistent
        })

        return jsonify(results)

    except Exception as e:
        app_logger.error(f"Unhandled exception in research endpoint: {str(e)}", exc_info=True)

        # Return an error response for unhandled exceptions
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return jsonify({
            "error": "An unexpected unhandled error occurred during processing",
            "details": str(e),
            "processing_time": f"{processing_time:.2f} seconds",
             # Include any partial data for debugging
            "partial_serp_data": locals().get('serp_data', {}),
            "partial_competitor_data": locals().get('competitor_data', {}),
            "partial_keyword_analysis": locals().get('keyword_analysis', {}),
            "partial_insights": locals().get('insights', {}),
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """
    Status endpoint to check if the server is running and components are initialized
    """
    api_key_status = "available" if os.getenv("GOOGLE_API_KEY") else "missing"

    # Check initialization status
    initialized = all([input_handler, serp_collector, content_analyzer, keyword_processor, insight_generator, result_renderer])

    components_status = {
        "input_handler": input_handler is not None,
        "serp_collector": serp_collector is not None,
        "content_analyzer": content_analyzer is not None,
        "keyword_processor": keyword_processor is not None,
        "insight_generator": insight_generator is not None,
        "result_renderer": result_renderer is not None,
        "core_dependencies": GEMINI_AVAILABLE and Agent is not None,
        "initialized_successfully": initialized
    }


    return jsonify({
        "status": "operational" if initialized else "partially_operational",
        "components": components_status,
        "api_key_status": api_key_status,
        "server_time": datetime.now().isoformat(),
        "version": "1.0.2",
        "message": "All components initialized." if initialized else "Core dependencies missing or components failed to initialize. Check logs."
    })


if __name__ == "__main__":
    # init_components() is called once on import at the top level now
    # check init_success status before running the app
    if not init_success:
         logger.error("Failed to initialize components. Exiting.")
         sys.exit(1) # Exit if initialization failed

    app_logger.info("Starting Deep Keyword Research API server...")
    # Using allow_unsafe_werkzeug=True for debug mode as per original app.py
    app.run(port=5000, debug=True)

