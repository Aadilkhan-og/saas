# Filename: src/app.py
import asyncio
import os
import logging
import sys
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Import necessary modules
from modules.input_handler import InputHandler
from modules.serp_collector import SerpCollector # Updated
from modules.content_analyzer import ContentAnalyzer # Updated
from modules.keyword_processor import KeywordProcessor # Updated
from modules.insight_generator import InsightGenerator # Updated
from modules.result_renderer import ResultRenderer

# Import dependencies checked during initialization
try:
    from src.modules.gemini_client import GeminiClient
except ImportError:
    GeminiClient = None

try:
    from browser_use import Agent # Still needed for ContentAnalyzer
except ImportError:
    Agent = None

# Import SerpAPI client to check availability
try:
    from serpapi import GoogleSearch # Used in SerpCollector, check here too
except ImportError:
    GoogleSearch = None


# Configure logging with proper encoding handling
def setup_logging():
    """Configure logging with proper encoding handling and formatting."""
    log_filename = f"keyword_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_level = logging.DEBUG # Set to INFO or WARNING for less verbose production logs

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear any existing handlers to avoid duplication in Flask debug mode restarts
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # Always log DEBUG to file
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Keep console less verbose
    # Custom formatter to handle emoji characters gracefully (from original)
    class EmojiFilter(logging.Formatter):
        """Custom formatter to handle emoji characters in logs"""
        def format(self, record):
            msg = super().format(record)
            # Replace emoji characters with text equivalents (can expand this list)
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
                # Add other emojis if needed based on browser-use/LLM output
            }
            for emoji, replacement in emoji_replacements.items():
                msg = msg.replace(emoji, replacement)
            # Ensure proper encoding for non-emoji special characters if needed
            return msg.encode('utf-8', 'replace').decode('utf-8')


    console_formatter = EmojiFilter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Also configure specific loggers from libraries if they are too chatty
    # logging.getLogger("serpapi").setLevel(logging.WARNING)
    # logging.getLogger("httpx").setLevel(logging.WARNING) # SerpAPI client might use httpx
    # logging.getLogger("httpcore").setLevel(logging.WARNING)


    return logger

# Set up logging early
logger = setup_logging()
app_logger = logging.getLogger("keyword_research.app")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    app_logger.info("Loaded environment variables from .env file")
except ImportError:
    app_logger.warning("python-dotenv not installed. Using existing environment variables.")
except Exception as e:
    app_logger.error(f"Error loading .env file: {str(e)}")


app = Flask(__name__)
CORS(app)

# Initialize components - hold instances here
input_handler = None
serp_collector = None
content_analyzer = None
keyword_processor = None
insight_generator = None
result_renderer = None

# Track initialization status
components_initialized = False
initialization_errors = []

def init_components():
    """Initialize all components with proper error handling and dependency checks."""
    global input_handler, serp_collector, content_analyzer, keyword_processor, insight_generator, result_renderer, components_initialized, initialization_errors

    initialization_errors = []
    components_initialized = False # Assume failure until all succeed

    app_logger.info("Attempting to initialize application components...")

    # --- Check Core Dependencies ---
    missing_deps = []
    if not Agent: missing_deps.append("browser_use")
    if not GoogleSearch: missing_deps.append("serpapi (google-search-results)")

    if missing_deps:
        error_msg = f"Missing core dependencies: {', '.join(missing_deps)}. Cannot initialize components."
        app_logger.error(error_msg)
        initialization_errors.append(error_msg)
        return False # Initialization failed

    # --- Check Required API Keys ---
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        error_msg = "GOOGLE_API_KEY environment variable not set. Gemini LLM features may not work correctly."
        app_logger.warning(error_msg) # Warning, as some parts might still work

    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_api_key:
        error_msg = "SERPAPI_API_KEY environment variable not set. SerpAPI collection will not work."
        app_logger.error(error_msg) # Error, as core data collection is critical
        initialization_errors.append(error_msg)
        # Decide if this is a fatal error. For a tool relying on SerpAPI, it likely is.
        return False # Initialization failed if SerpAPI key is missing

    # --- Initialize Components ---
    try:
        input_handler = InputHandler()
        app_logger.info("InputHandler initialized.")

        serp_collector = SerpCollector() # Updated to use SerpAPI
        if not serp_collector.serpapi_available:
             error_msg = "SerpCollector failed to initialize properly (SerpAPI not available/configured)."
             app_logger.error(error_msg)
             initialization_errors.append(error_msg)
             return False # Fatal if collector fails


        content_analyzer = ContentAnalyzer() # Updated, still uses browser-use
        if not content_analyzer.agent_available:
             error_msg = "ContentAnalyzer failed to initialize properly (browser-use/LLM not available)."
             app_logger.warning(error_msg) # Warning, content analysis is secondary data


        keyword_processor = KeywordProcessor() # Updated for scoring/clustering
        # Check if NLP is available (warning if not)
        if not keyword_processor.sentence_model:
             app_logger.warning("NLP model not loaded for KeywordProcessor. Clustering will be basic.")


        insight_generator = InsightGenerator() # Updated for AI insights
        if not insight_generator.llm_available:
             app_logger.warning("InsightGenerator LLM not available. Insights will be basic/static.")


        result_renderer = ResultRenderer() # Will need to handle new data structure
        app_logger.info("ResultRenderer initialized.")

        # If we reached here, all critical components initialized
        components_initialized = True
        app_logger.info("All application components initialized successfully.")
        return True

    except Exception as e:
        error_msg = f"An unexpected error occurred during component initialization: {str(e)}"
        app_logger.error(error_msg, exc_info=True)
        initialization_errors.append(error_msg)
        components_initialized = False
        return False # Initialization failed


# Attempt to initialize components on startup
init_success = init_components()


@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/api/research', methods=['POST'])
async def research():
    """
    Main endpoint for the deep keyword research feature.
    Orchestrates calls to modules.
    """
    start_time = datetime.now()
    app_logger.info(f"Received research request at {start_time}")

    # Check if components initialized successfully on startup
    if not components_initialized:
        app_logger.error("Research request received but components failed to initialize on startup.")
        return jsonify({"error": "Server components not initialized correctly. Check server logs for details.", "initialization_errors": initialization_errors}), 500

    # Ensure key components required for this specific request are available
    # SerpCollector is critical for getting data from SerpAPI
    if not serp_collector or not serp_collector.serpapi_available:
         app_logger.error("SerpCollector is not available or SerpAPI not configured.")
         return jsonify({"error": "Keyword data collection service is not available."}), 500

    # ContentAnalyzer is important, but research can proceed without it (partial data)
    content_analysis_possible = content_analyzer and content_analyzer.agent_available
    if not content_analysis_possible:
         app_logger.warning("Content analysis will be skipped due to unavailable component.")

    # KeywordProcessor and InsightGenerator are needed for analysis and insights
    if not keyword_processor or not insight_generator:
         app_logger.error("Keyword processing or insight generation components are missing.")
         return jsonify({"error": "Analysis components are not available."}), 500

    try:
        # Get request data
        data = request.get_json()
        app_logger.debug(f"Received request data: {data}")

        # --- Step 1: Process Input ---
        processed_input = {}
        try:
            processed_input = input_handler.process_input(data)
            app_logger.info(f"Processed input data. Seed keyword: {processed_input.get('seed_keyword')}, Total keywords: {len(processed_input.get('keywords', []))}")
        except ValueError as e:
            app_logger.error(f"Input validation failed: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
             app_logger.error(f"Unexpected error during input processing: {str(e)}", exc_info=True)
             return jsonify({"error": "An unexpected error occurred during input processing."}), 500


        keywords_to_research = processed_input.get('keywords', [])
        if not keywords_to_research:
             app_logger.warning("No keywords provided after input processing.")
             return jsonify({"error": "No valid keywords provided for research."}), 400


        # --- Step 2: Collect SERP data (using SerpAPI) ---
        # Max results per keyword for SERP data collection
        serp_max_results_per_kw = 10 # Can make this configurable later

        serp_data = {} # Initialize with empty dict
        try:
            # Use a reasonable timeout for SERP collection per keyword
            # The SerpCollector itself handles iteration and internal timeouts/retries per API call
            app_logger.info(f"Starting SERP data collection for {len(keywords_to_research)} keywords.")
            # serp_collector.collect_serp_data is already async and handles looping/delays internally
            serp_data = await serp_collector.collect_serp_data(
                keywords_to_research,
                max_results=serp_max_results_per_kw
                )
            app_logger.info(f"SERP data collection finished.")
            app_logger.debug(f"Collected SERP data structure keys: {serp_data.keys() if serp_data else 'None'}")
            app_logger.info(f"Keywords with SERP results: {len(serp_data.get('serp_data', {}))}, Total top URLs collected: {len(serp_data.get('top_urls', []))}")

            # Basic check if any useful SERP data was collected
            if not serp_data or not serp_data.get("serp_data"):
                 app_logger.warning("SERP data collection completed but returned no results.")
                 # Decide if this is a fatal error or if we proceed with partial data
                 # Proceeding with partial data (empty SERP) allows subsequent steps to run, though results will be minimal
                 pass # Allow to continue

        except Exception as e:
            app_logger.error(f"Error during SERP data collection: {str(e)}", exc_info=True)
            # Continue with potentially empty serp_data rather than failing completely
            serp_data = serp_collector._create_empty_results_structure(keywords_to_research)
            serp_data["error"] = f"SERP data collection failed: {str(e)}" # Add error flag
            app_logger.warning("Continuing analysis with empty SERP data due to error.")


        # --- Step 3: Analyze competitor content ---
        competitor_data = {} # Initialize with empty dict
        urls_to_analyze = serp_data.get('top_urls', []) # Use URLs from SERP data
        content_analysis_max_urls = 5 # Max URLs for content analysis (can be configured)

        if urls_to_analyze and content_analysis_possible:
            try:
                 app_logger.info(f"Starting competitor content analysis for {len(urls_to_analyze)} URLs (max {content_analysis_max_urls}).")
                 # content_analyzer.analyze_content is already async and handles iteration/timeouts per URL
                 competitor_data = await content_analyzer.analyze_content(
                     urls_to_analyze,
                     max_urls=content_analysis_max_urls
                     )
                 app_logger.info(f"Competitor content analysis finished.")
                 app_logger.info(f"URLs successfully analyzed: {len(competitor_data.get('analyzed_urls', []))}")

            except Exception as e:
                app_logger.error(f"Error during competitor content analysis: {str(e)}", exc_info=True)
                # Continue with potentially empty competitor_data
                competitor_data = {
                    "analyzed_urls": [], "content_analysis": {}, "common_themes": [],
                    "content_types": {}, "heading_structure": {}, "content_length": {},
                     "readability_scores": {}, # Ensure new keys exist
                    "summary": {}
                }
                competitor_data["error"] = f"Content analysis failed: {str(e)}" # Add error flag
                app_logger.warning("Continuing analysis without complete competitor data due to error.")

        else:
            if not urls_to_analyze:
                 app_logger.warning("No top URLs collected from SERP, skipping content analysis.")
            if not content_analysis_possible:
                 app_logger.warning("Content analysis component is not available, skipping analysis.")

            # Ensure empty structure if analysis is skipped
            competitor_data = {
                "analyzed_urls": [], "content_analysis": {}, "common_themes": [],
                "content_types": {}, "heading_structure": {}, "content_length": {},
                "readability_scores": {},
                "summary": {}
            }
            if not content_analysis_possible:
                 competitor_data["error"] = "Content analysis component unavailable."
            elif not urls_to_analyze:
                 competitor_data["error"] = "No URLs from SERP to analyze."


        # --- Step 4: Process keywords (classification, clustering, scoring) ---
        keyword_analysis = {} # Initialize with empty dict
        try:
            app_logger.info("Starting keyword processing (intent, clustering, scoring).")
            keyword_analysis = keyword_processor.process_keywords(
                keywords_to_research, # Process all input keywords
                serp_data, # Pass collected serp_data (includes SV/KD now)
                competitor_data # Pass competitor data
            )
            app_logger.info(f"Keywords processed. Intents classified: {len(keyword_analysis.get('intent_classification', {}))}, Clusters: {len(keyword_analysis.get('clusters', []))}, Scored keywords: {len(keyword_analysis.get('keyword_scores', {}))}")

        except Exception as e:
            app_logger.error(f"Error during keyword processing: {str(e)}", exc_info=True)
            # Continue with default data
            keyword_analysis = {
                "intent_classification": {kw: "unknown" for kw in keywords_to_research}, # Add unknown intent for all input keywords
                "clusters": [],
                "keyword_scores": {kw: {"difficulty": 50, "opportunity": 50, "score": 50} for kw in keywords_to_research}, # Add default scores
                "question_keywords": [],
                "error": f"Keyword processing failed: {str(e)}"
            }
            app_logger.warning("Continuing analysis with default keyword analysis data due to error.")


        # --- Step 5: Generate insights (AI and rule-based) ---
        insights = {} # Initialize with empty dict
        # Pass ALL relevant data to the insight generator
        try:
            app_logger.info("Starting insight generation.")
            insights = await insight_generator.generate_insights(
                processed_input, # User input data
                keyword_analysis, # Processed keyword data
                serp_data, # Comprehensive SerpAPI data
                competitor_data # Detailed competitor analysis
            )
            app_logger.info("Insight generation finished.")
            # app_logger.debug(f"Generated insights structure keys: {insights.keys()}") # Log insight keys


        except Exception as e:
            app_logger.error(f"Error during insight generation: {str(e)}", exc_info=True)
            # Continue with default insights
            insights = {
                "content_opportunities": [],
                "serp_feature_insights": [],
                "competitive_landscape_summary": {},
                "keyword_recommendations": [],
                "topic_clusters": keyword_analysis.get("clusters", []), # Use clusters from processor if available
                "intent_distribution": keyword_analysis.get("intent_distribution", {}), # Use intent from processor if available
                "content_blueprints": {}, # Blueprints likely failed if insight generation had error
                "summary": {"overall_summary": f"Analysis complete. Error generating complete insights: {str(e)}"}, # Indicate error in summary
                "error": f"Insight generation failed: {str(e)}"
            }
            app_logger.warning("Returning partial insights due to error.")


        # --- Step 6: Render results (Prepare the final JSON structure) ---
        results = {} # Initialize with empty dict
        try:
             # Pass ALL collected/processed/generated data to the renderer
             # The renderer will structure this into the final output format
            results = result_renderer.render_results(
                processed_input,
                keyword_analysis,
                serp_data, # Pass comprehensive SerpAPI data for rendering
                competitor_data, # Pass detailed competitor data for rendering
                insights, # Pass generated insights
                serp_data, # Pass again as raw_serp_data for renderer's detailed view
                competitor_data # Pass again as raw_competitor_data for renderer's detailed view
            )
            app_logger.info("Results structure prepared successfully.")

        except Exception as e:
            app_logger.error(f"Error rendering final results structure: {str(e)}", exc_info=True)
            # Return a simplified response in case of rendering error
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            return jsonify({
                "error": "Error preparing final results structure for display.",
                "details": str(e),
                "summary": {
                    "title": f"Keyword Research: {processed_input.get('seed_keyword', 'Unknown')}",
                    "date": end_time.strftime("%Y-%m-%d"),
                    "keyword_count": len(processed_input.get('keywords', [])),
                    "processing_time": f"{processing_time:.2f} seconds",
                    "insight_summary": "An error occurred while organizing the final results.",
                    "top_opportunities": [] # Cannot guarantee this list on rendering error
                },
                 # Include the potentially incomplete data for debugging
                "partial_serp_data": serp_data,
                "partial_competitor_data": competitor_data,
                "partial_keyword_analysis": keyword_analysis,
                "partial_insights": insights,

            }), 500


        # Calculate and log processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        app_logger.info(f"Research request completed in {processing_time:.2f} seconds")

        # Add processing metadata to results
        if 'metadata' not in results:
             results['metadata'] = {}

        results["metadata"].update({
            "processing_time_seconds": processing_time,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            # Use counts from the actual data collected/analyzed
            "keywords_input": len(keywords_to_research),
            "serp_keywords_processed": len(serp_data.get('serp_data', {})), # Count keywords for which we got SERP data
            "urls_analyzed": len(competitor_data.get('analyzed_urls', [])), # Count URLs successfully analyzed
            "version": "1.0.3" # Update version
        })

        # Include component initialization errors in the response metadata for debugging if any occurred
        if initialization_errors:
            results["metadata"]["initialization_errors"] = initialization_errors


        return jsonify(results)

    except Exception as e:
        app_logger.error(f"Unhandled exception in research endpoint: {str(e)}", exc_info=True)

        # Return a generic error response for unhandled exceptions
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Attempt to include any data collected up to the point of unhandled error
        return jsonify({
            "error": "An unexpected unhandled error occurred during processing.",
            "details": str(e),
            "processing_time": f"{processing_time:.2f} seconds",
             # Include any partial data for debugging if locals() are accessible (depends on scope)
            "partial_processed_input": locals().get('processed_input', {}),
            "partial_serp_data": locals().get('serp_data', {}),
            "partial_competitor_data": locals().get('competitor_data', {}),
            "partial_keyword_analysis": locals().get('keyword_analysis', {}),
            "partial_insights": locals().get('insights', {}),
             "initialization_errors": initialization_errors # Include startup errors too
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """
    Status endpoint to check if the server is running and components are initialized.
    """
    # Check API key statuses
    google_api_key_status = "available" if os.getenv("GOOGLE_API_KEY") else "missing"
    serpapi_api_key_status = "available" if os.getenv("SERPAPI_API_KEY") else "missing"

    # Check initialization status of components
    # Check if the *instances* exist
    components_status = {
        "input_handler": input_handler is not None,
        "serp_collector": serp_collector is not None,
        "content_analyzer": content_analyzer is not None,
        "keyword_processor": keyword_processor is not None,
        "insight_generator": insight_generator is not None,
        "result_renderer": result_renderer is not None,
        # Check underlying dependencies/readiness
        "core_dependencies_check": {
             "gemini_client (google-generativeai, langchain-google-genai)": GeminiClient is not None,
             "browser_use": Agent is not None, # Still needed for ContentAnalyzer
             "serpapi_client (google-search-results)": GoogleSearch is not None, # Needed for SerpCollector
        },
        "component_readiness": { # Check internal readiness flags from __init__
             "serp_collector_available": serp_collector.serpapi_available if serp_collector else False,
             "content_analyzer_available": content_analyzer.agent_available if content_analyzer else False,
             "keyword_processor_nlp_available": keyword_processor.sentence_model is not None if keyword_processor else False,
             "insight_generator_llm_available": insight_generator.llm_available if insight_generator else False,
        },
        "initialized_successfully": components_initialized, # Overall flag
        "initialization_errors": initialization_errors # List any errors from init
    }

    overall_status = "operational" if components_initialized else "partially_operational"
    if not serp_collector or not serp_collector.serpapi_available:
         overall_status = "critical_failure" # Cannot collect core data

    message = "All core components initialized and available." if components_initialized else "Server is running but some components failed to initialize or dependencies are missing. Check 'component_readiness' and 'initialization_errors'."
    if overall_status == "critical_failure":
        message = "Server is running but critical data collection component failed to initialize. Check logs and configuration (SerpAPI)."

    return jsonify({
        "status": overall_status,
        "message": message,
        "api_key_status": {
            "GOOGLE_API_KEY": google_api_key_status,
            "SERPAPI_API_KEY": serpapi_api_key_status,
        },
        "components": components_status,
        "server_time": datetime.now().isoformat(),
        "version": "1.0.3", # Update version
    })


if __name__ == "__main__":
    # init_components() is called once on import at the top level now
    # check init_success status before running the app
    # We will allow the app to run even with warnings (e.g., no content analyzer)
    # but exit on critical failure (e.g., no SerpAPI key/client)

    # The init_components already sets the `components_initialized` flag and `initialization_errors`
    # The /status endpoint uses these. We don't necessarily need to exit here if it's just a warning.
    # The research endpoint will perform checks for critical components.

    app_logger.info("Starting Deep Keyword Research API server...")
    # Using allow_unsafe_werkzeug=True for debug mode
    app.run(port=5000, debug=True)