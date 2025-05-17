# Filename: src/modules/serp_collector.py
import os
import re
import json
import logging
import asyncio
# Removed browser_use dependency for SERP collection
# try:
#     from browser_use import Agent, AgentHistoryList
# except ImportError:
#     Agent = None
#     AgentHistoryList = None

# Import SerpAPI client or library for making requests
# Assuming you have the SerpAPI Python client installed: pip install google-search-results
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None
    logging.error("SerpAPI Python client 'google-search-results' not installed. SERP collection will not work.")


# Import ChatOpenAI here as well, though it's less involved in *collection* now
# Removed ChatOpenAI import and usage; SerpCollector does not require an LLM

# Create module logger
logger = logging.getLogger("keyword_research.serp_collector")

class SerpCollector:
    """
    Collects search engine results page (SERP) data using SerpAPI
    """

    def __init__(self):
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        if not self.serpapi_api_key:
            logger.error("SERPAPI_API_KEY environment variable not set. SerpAPI collection will not work.")
            self.serpapi_available = False
        elif not GoogleSearch:
             self.serpapi_available = False
        else:
            self.serpapi_available = True
            logger.info("SerpAPI client initialized.")

        # LLM is not strictly needed for collection itself now, but keep if needed elsewhere later
        # Removed OpenAI/LLM initialization; SerpCollector is now LLM-agnostic


    async def collect_serp_data(self, keywords, max_results=10):
        """
        Collect SERP data for the provided keywords using SerpAPI

        Args:
            keywords (list): List of keywords to research
            max_results (int): Maximum number of organic results to collect per keyword

        Returns:
            dict: Collected SERP data including SV and KD
        """
        logger.info(f"Collecting SERP data for {len(keywords)} keywords using SerpAPI")

        if not self.serpapi_available:
            logger.error("SerpAPI is not available. Skipping SERP collection.")
            return self._create_empty_results_structure(keywords)


        # Initialize results dictionary
        results = {
            "keywords": keywords,
            "serp_data": {}, # Stores structured organic results per keyword
            "features": {}, # Stores identified features per keyword
            "paa_questions": {}, # Stores PAA questions per keyword
            "related_searches": {}, # Stores related searches per keyword
            "top_urls": [], # Stores top URLs across all keywords
            "search_volume": {}, # Stores Search Volume per keyword (from API)
            "keyword_difficulty": {} # Stores Keyword Difficulty per keyword (from API)
        }

        # Iterate through ALL keywords (assuming 3-keyword limit fix is done)
        for i, keyword in enumerate(keywords):
            try:
                logger.info(f"Processing keyword {i+1}/{len(keywords)}: {keyword}")

                # Add small delay between requests to avoid hitting API rate limits aggressively
                if i > 0:
                    await asyncio.sleep(1) # Shorter delay with API calls usually

                # Collect SERP data for this keyword using SerpAPI
                keyword_data = await self._collect_keyword_serp(keyword, max_results)

                # Store results, ensuring keys exist and are lists/dicts as expected
                if keyword_data:
                    results["serp_data"][keyword] = keyword_data.get("organic_results", []) if isinstance(keyword_data.get("organic_results"), list) else []
                    results["features"][keyword] = keyword_data.get("detected_features", []) if isinstance(keyword_data.get("detected_features"), list) else []
                    results["paa_questions"][keyword] = keyword_data.get("paa_questions", []) if isinstance(keyword_data.get("paa_questions"), list) else []
                    results["related_searches"][keyword] = keyword_data.get("related_searches", []) if isinstance(keyword_data.get("related_searches"), list) else []

                    # Store SV and KD if available from the API
                    if keyword_data.get("search_parameters", {}).get("engine") == "google_keywords":
                        # SerpAPI's Google Keywords API returns SV/KD directly in the main response
                        results["search_volume"][keyword] = keyword_data.get("search_volume")
                        results["keyword_difficulty"][keyword] = keyword_data.get("keyword_difficulty")
                        logger.info(f"Fetched SV/KD for '{keyword}': SV={results['search_volume'][keyword]}, KD={results['keyword_difficulty'][keyword]}")
                    elif keyword_data.get("serpapi_pagination"):
                         # If using standard Google Search API, SV/KD might be in a separate tool or not directly in SERP result
                         # Depending on your SerpAPI plan and desired approach, you might need a separate call for SV/KD
                         # For now, we'll assume SV/KD comes from a specific API type or is handled downstream if not here.
                         # Let's add a placeholder check if SV/KD were somehow attached (less common for standard SERP API)
                         sv_kd_data = keyword_data.get("keyword_info") # Example key, check SerpAPI docs
                         if sv_kd_data:
                              results["search_volume"][keyword] = sv_kd_data.get("search_volume")
                              results["keyword_difficulty"][keyword] = sv_kd_kd_data.get("difficulty")


                    # Add top URLs to the overall list (from organic results)
                    top_urls = []
                    # Get top URLs from the results list for this keyword
                    for result in results["serp_data"][keyword]: # Use all results fetched, not just top 3 initially
                        url = result.get("link", "") # SerpAPI uses 'link' for URL
                        # Clean and validate URLs
                        url = self._clean_url(url)
                        if url:
                            top_urls.append(url)

                    # Limit the number of URLs added to the overall list to avoid excessive content analysis later
                    results["top_urls"].extend(top_urls[:min(max_results, len(top_urls))]) # Add up to max_results URLs


                    logger.info(f"Successfully collected SERP data for keyword: {keyword}. Organic results: {len(results['serp_data'][keyword])}, PAA: {len(results['paa_questions'][keyword])}")
                else:
                     logger.warning(f"No data returned from SerpAPI for keyword: {keyword}")
                     # Ensure empty entries for this keyword if no data was returned
                     results["serp_data"][keyword] = []
                     results["features"][keyword] = []
                     results["paa_questions"][keyword] = []
                     results["related_searches"][keyword] = []
                     results["search_volume"][keyword] = None
                     results["keyword_difficulty"][keyword] = None


            except Exception as e:
                logger.error(f"Failed to collect SERP data for keyword '{keyword}' using SerpAPI: {str(e)}", exc_info=True)
                # Ensure empty entries for this keyword if an error occurred
                results["serp_data"][keyword] = []
                results["features"][keyword] = []
                results["paa_questions"][keyword] = []
                results["related_searches"][keyword] = []
                results["search_volume"][keyword] = None
                results["keyword_difficulty"][keyword] = None


        # Remove duplicate URLs from the overall list
        results["top_urls"] = list(dict.fromkeys(results["top_urls"]))
        logger.info(f"Collected {len(results['top_urls'])} unique top URLs across all keywords for potential content analysis.")

        return results

    async def _collect_keyword_serp(self, keyword, max_results):
        """
        Collect SERP data for a single keyword using SerpAPI.
        Note: SerpAPI library handles async internally if used with await.

        Args:
            keyword (str): Keyword to research
            max_results (int): Maximum number of organic results to collect

        Returns:
            dict: Raw data from SerpAPI response, or None on failure
        """
        logger.debug(f"Calling SerpAPI for keyword: {keyword}")
        if not self.serpapi_available:
            logger.error("SerpAPI not available for _collect_keyword_serp.")
            return None

        try:
            # SerpAPI parameters (customize as needed)
            params = {
                "engine": "google",          # Use Google Search engine
                "q": keyword,                # The search query
                "api_key": self.serpapi_api_key,
                "num": max_results,          # Number of organic results (10 is max per page usually)
                "hl": "en",                  # Host Language
                "gl": "us",                  # Geo Location (customize based on user input/default)
                 # Add other parameters as needed: location, device, etc.
                 # "location": "United States", # Example: specify a location
                 # "device": "mobile",        # Example: specify device type
            }

            # --- Option 1: Using SerpAPI's GoogleSearch client ---
            # The client can handle the request and return JSON
            # It might have async support if used with await, check library documentation
            search = GoogleSearch(params)
            # Use get_dict() for the JSON response
            # Use get_json() for the JSON string
            # Asynchronous execution with SerpAPI client might require specific methods or wrappers if get_dict/get_json are synchronous.
            # A simple await might work if the library supports it implicitly with async def.
            # If not, you might need to run it in a thread or use a different http client like aiohttp.

            # Assuming get_dict() is awaitable or can be run in a thread
            # Let's try a simple await first, if it fails, threading might be needed.
            # Alternatively, use aiohttp for the request if SerpAPI client is synchronous.
            # Example using get_dict() - assuming it's compatible with async or run in executor
            # raw_results = await search.get_dict() # If get_dict is awaitable
            # If get_dict is synchronous, run in executor:
            loop = asyncio.get_running_loop()
            raw_results = await loop.run_in_executor(None, search.get_dict)

            # --- Option 2: Using aiohttp for direct HTTP request (More control, manual URL/parsing) ---
            # import aiohttp
            # api_url = "https://serpapi.com/search"
            # async with aiohttp.ClientSession() as session:
            #     async with session.get(api_url, params=params) as response:
            #         raw_results = await response.json()
            #         response.raise_for_status() # Raise an exception for bad status codes

            logger.debug(f"SerpAPI call successful for keyword: {keyword}")
            # logger.debug(f"Raw SerpAPI results for '{keyword}': {json.dumps(raw_results, indent=2)}") # Log full raw results for debugging

            # Process raw results into the desired structure
            processed_data = self._process_serpapi_results(raw_results)
            return processed_data

        except Exception as e:
            logger.error(f"Error calling SerpAPI for keyword '{keyword}': {str(e)}", exc_info=True)
            return None

    def _process_serpapi_results(self, raw_results):
        """
        Processes the raw JSON response from SerpAPI into a consistent structure.

        Args:
            raw_results (dict): The raw JSON dictionary from SerpAPI.

        Returns:
            dict: Processed data with standardized keys, or empty structure if input is invalid.
        """
        if not isinstance(raw_results, dict):
            logger.error("Invalid raw_results received for processing (not a dictionary).")
            return self._create_empty_serp_data_structure()

        processed = self._create_empty_serp_data_structure()

        # Extract Organic Results
        processed["organic_results"] = []
        if "organic_results" in raw_results and isinstance(raw_results["organic_results"], list):
            for result in raw_results["organic_results"]:
                if isinstance(result, dict):
                    processed["organic_results"].append({
                        "position": result.get("position"),
                        "title": result.get("title"),
                        "link": result.get("link"), # SerpAPI uses 'link'
                        "snippet": result.get("snippet"), # SerpAPI uses 'snippet'
                         # Add other relevant fields from organic_results if needed (e.g., site_link)
                    })

        # Extract Featured Snippet
        if "featured_snippet" in raw_results and isinstance(raw_results["featured_snippet"], dict):
             processed["featured_snippet"] = {
                "title": raw_results["featured_snippet"].get("title"),
                "link": raw_results["featured_snippet"].get("link"), # SerpAPI uses 'link'
                "snippet": raw_results["featured_snippet"].get("snippet"), # SerpAPI uses 'snippet'
                "source": raw_results["featured_snippet"].get("source") # Source can be useful
             }


        # Extract People Also Ask
        processed["paa_questions"] = []
        if "related_questions" in raw_results and isinstance(raw_results["related_questions"], list):
            for question_block in raw_results["related_questions"]:
                if isinstance(question_block, dict) and "question" in question_block:
                    processed["paa_questions"].append(question_block["question"])
                    # You might also want to extract the answer if needed, check SerpAPI structure

        # Extract Related Searches
        processed["related_searches"] = []
        if "related_searches" in raw_results and isinstance(raw_results["related_searches"], list):
            for search_term in raw_results["related_searches"]:
                 processed["related_searches"].append(search_term.get("query") or search_term.get("snippet") or search_term.get("title")) # Handle different possible keys

        # Detect other prominent SERP Features present
        # This is a simplified detection. A more comprehensive approach would check for keys
        # like "shopping_results", "videos", "images", "local_results", "knowledge_graph", etc.
        detected_features = []
        if "featured_snippet" in raw_results and raw_results["featured_snippet"]:
             detected_features.append("featured_snippet")
        if "related_questions" in raw_results and raw_results["related_questions"]:
             detected_features.append("people_also_ask")
        if "videos" in raw_results and raw_results["videos"]:
             detected_features.append("video_results")
        if "images" in raw_results and raw_results["images"]:
             detected_features.append("image_pack")
        if "shopping_results" in raw_results and raw_results["shopping_results"]:
             detected_features.append("shopping_results")
        if "local_results" in raw_results and raw_results["local_results"]:
            detected_features.append("local_pack")
        if "knowledge_graph" in raw_results and raw_results["knowledge_graph"]:
             detected_features.append("knowledge_panel")

        processed["detected_features"] = detected_features

        # Include raw search parameters and potentially SV/KD if present in the response
        processed["search_parameters"] = raw_results.get("search_parameters")
        # Note: SV/KD might be in the main response body for google_keywords engine,
        # or require a separate API call depending on SerpAPI product and plan.
        # We will assume they are pulled in the main collect_serp_data loop if available there.

        return processed

    def _create_empty_serp_data_structure(self):
        """Create empty structure for processed SerpAPI data."""
        return {
            "organic_results": [],
            "featured_snippet": None,
            "paa_questions": [],
            "related_searches": [],
            "detected_features": [],
            "search_parameters": None,
            # Add placeholders for SV/KD if they were expected here
        }

    def _create_empty_results_structure(self, keywords):
         """Create empty overall results structure for error cases"""
         return {
            "keywords": keywords,
            "serp_data": {},
            "features": {},
            "paa_questions": {},
            "related_searches": {},
            "top_urls": [],
            "search_volume": {},
            "keyword_difficulty": {}
        }


    def _clean_url(self, url):
        """
        Clean and normalize a URL - Keep existing cleaning logic

        Args:
            url (str): URL to clean

        Returns:
            str: Cleaned URL
        """
        # Skip if empty
        if not url:
            return ""

        # Handle various URL formatting issues
        url = str(url)  # Ensure string type

        # Remove common formatting issues like quotes or escaped quotes
        url = url.replace('"', '').replace("'", '')
        # Fix common protocol issues if present
        url = url.replace('""https:', 'https:').replace('""http:', 'http:')
        url = url.replace('"https":', 'https:').replace('"http":', 'http:')


        # Ensure URL has proper scheme
        if url and not url.startswith(('http://', 'https://')):
             # Check if it's just a domain name
            if '.' in url and '/' not in url:
                url = 'https://' + url # Assume https for domain names
            else:
                # Otherwise, it might be a malformed path or other issue, return empty
                logger.warning(f"URL '{url}' does not start with http/https and is not a simple domain. Skipping.")
                return ""


        return url.strip()