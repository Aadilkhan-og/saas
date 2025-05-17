# Filename: src/modules/serp_collector.py
import os
import re
import json
import logging
import asyncio
from browser_use import Agent, AgentHistoryList
from modules.gemini_client import GeminiClient

# Import to check availability consistently
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
# Import AgentHistoryList and ActionResult to check types



# Create module logger
logger = logging.getLogger("keyword_research.serp_collector")

class SerpCollector:
    """
    Collects search engine results page (SERP) data using browser-use
    """

    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        try:
            # Check if Agent and Gemini are available
            if not Agent or not GEMINI_AVAILABLE:
                 logger.error("browser_use or Gemini modules not installed. SERP collection will not work.")
                 self.llm = None
                 return
            gemini_client = GeminiClient(api_key=self.google_api_key, model="gemini-2.0-flash")
            self.llm = gemini_client.get_langchain_llm()
            logger.info("Gemini initialized for SerpCollector")
        except Exception as e:
            logger.error(f"Error initializing Gemini: {str(e)}")
            self.llm = None


    async def collect_serp_data(self, keywords, max_results=10):
        """
        Collect SERP data for the provided keywords

        Args:
            keywords (list): List of keywords to research
            max_results (int): Maximum number of results to collect per keyword

        Returns:
            dict: Collected SERP data
        """
        logger.info(f"Collecting SERP data for {len(keywords)} keywords")

        # Check if LLM or Agent are initialized
        if not self.llm or not Agent:
            logger.error("LLM or Agent not initialized for SerpCollector. Skipping SERP collection.")
            return {
                "keywords": keywords,
                "serp_data": {},
                "features": {},
                "paa_questions": {},
                "related_searches": {},
                "top_urls": []
            }


        # Initialize results dictionary
        results = {
            "keywords": keywords,
            "serp_data": {},
            "features": {},
            "paa_questions": {},
            "related_searches": {},
            "top_urls": []
        }

        # Process each keyword
        # Limiting to 3 keywords for demo in the original code, keeping that for now
        for i, keyword in enumerate(keywords):
            try:
                logger.info(f"Processing keyword {i+1}/{min(len(keywords), 3)}: {keyword}")

                # Add small delay between requests to avoid rate limiting
                if i > 0:
                    await asyncio.sleep(2)

                # Collect SERP data for this keyword
                keyword_data = await self._collect_keyword_serp(keyword, max_results)

                # Store results, ensuring keys exist and are lists/dicts as expected
                results["serp_data"][keyword] = keyword_data.get("results", []) if isinstance(keyword_data.get("results"), list) else []
                results["features"][keyword] = keyword_data.get("features", []) if isinstance(keyword_data.get("features"), list) else []
                results["paa_questions"][keyword] = keyword_data.get("paa_questions", []) if isinstance(keyword_data.get("paa_questions"), list) else []
                results["related_searches"][keyword] = keyword_data.get("related_searches", []) if isinstance(keyword_data.get("related_searches"), list) else []


                # Add top URLs to the overall list
                top_urls = []
                # Get top 3 URLs from the results list for this keyword
                for result in results["serp_data"][keyword][:3]:
                    url = result.get("url", "")
                    # Clean URLs - remove quotes if present and validate structure
                    url = self._clean_url(url)
                    if url:
                        top_urls.append(url)

                results["top_urls"].extend(top_urls)

                logger.info(f"Successfully collected SERP data for keyword: {keyword}")

            except Exception as e:
                logger.error(f"Failed to collect SERP data for keyword '{keyword}': {str(e)}", exc_info=True)
                # Ensure empty entries for this keyword if an error occurred
                results["serp_data"][keyword] = []
                results["features"][keyword] = []
                results["paa_questions"][keyword] = []
                results["related_searches"][keyword] = []


        # Remove duplicate URLs
        results["top_urls"] = list(dict.fromkeys(results["top_urls"]))
        logger.info(f"Collected {len(results['top_urls'])} unique top URLs across all keywords")

        return results

    async def _collect_keyword_serp(self, keyword, max_results):
        """
        Collect SERP data for a single keyword

        Args:
            keyword (str): Keyword to research
            max_results (int): Maximum number of results to collect

        Returns:
            dict: SERP data for the keyword, or an empty structure on failure
        """
        # Check if LLM and Agent are initialized before creating task
        if not self.llm or not Agent:
            logger.error("LLM or Agent not initialized for SerpCollector. Cannot collect SERP for keyword.")
            return self._create_empty_serp_data()


        task = f"""
        Search for '{keyword}' and extract the following information accurately:

        1.  The top {max_results} organic search results. For each result, get its exact title, the full URL, and the description snippet.
        2.  Any Featured Snippet present, including its title, URL, and content.
        3.  All questions listed in the 'People Also Ask' section and their answers.
        4.  All suggestions listed in the 'Related Searches' section shown towards the bottom of the page.
        5.  List any other notable SERP features visible (e.g., Video Carousel, Image Pack, Shopping Results, Knowledge Panel, Top Stories).

        Return the results EXCLUSIVELY as a single, valid JSON object. Ensure all property names and string values within the JSON are correctly formatted and escaped. The JSON structure must be EXACTLY:
        {{
            "results": [
                {{"position": 1, "title": "Result Title", "url": "https://example.com/page", "description": "Result description snippet..."}},
                // ... up to {max_results} results ...
            ],
            "featured_snippet": {{"title": "Snippet Title", "url": "https://example.com/snippet-source", "content": "The content of the featured snippet..."}} or null if not found,
            "paa_questions": ["Question 1?", "Question 2?", ...],
            "related_searches": ["Related search term 1", "Related search term 2", ...],
            "features": ["video_carousel", "people_also_ask", "featured_snippet", "image_pack", "shopping_results", ...] // List identified features
        }}

        Do NOT include any other text, markdown formatting (like ```json```), or explanation outside the single JSON object. If a section (like featured_snippet, paa_questions, related_searches, features) is not present on the SERP, return it as null (for featured_snippet) or an empty array (for lists).
        """


        # Configure the agent with retry mechanism
        max_retries = 3 # Increased retries
        for attempt in range(max_retries + 1):
            try:
                # Create a browser-use agent - REMOVE browser_config
                agent = Agent(
                    task=task,
                    llm=self.llm,
                    # browser_config={"headless": True} # REMOVED invalid argument
                )

                # Run the agent
                logger.info(f"Running search agent for keyword (attempt {attempt+1}/{max_retries+1}): {keyword}")
                result = await agent.run()
                logger.info(f"Search attempt completed for keyword: {keyword}") # Only log the keyword, not the full result


                # Parse the result, prioritizing extraction from ActionResult objects
                serp_data = self._extract_json_from_agent_result(result)

                # Log the extracted data for debugging
                logger.debug(f"Extracted SERP data for '{keyword}': {serp_data}")

                # Basic validation: check if 'results' key exists and is a list and is not empty
                if isinstance(serp_data, dict) and isinstance(serp_data.get("results"), list) and serp_data.get("results"):
                     logger.info(f"Successfully extracted main SERP results data on attempt {attempt+1} for keyword: {keyword}")
                     # Add the original keyword to the data for context downstream if needed
                     serp_data['original_keyword'] = keyword
                     return serp_data
                else:
                    # Log a warning if the expected data structure wasn't found
                    if isinstance(serp_data, dict) and not serp_data.get("results"):
                         logger.warning(f"Agent result for '{keyword}' was a dictionary but 'results' list was empty or missing on attempt {attempt+1}/{max_retries+1}. Retrying...")
                    else:
                         logger.warning(f"Agent result for '{keyword}' did not return a dictionary with a 'results' list structure on attempt {attempt+1}/{max_retries+1}. Retrying...")


            except Exception as e:
                logger.error(f"Error running search agent for '{keyword}' (attempt {attempt+1}/{max_retries+1}): {str(e)}", exc_info=True)
                # Error occurred before or during agent run, retry

            # Wait a bit before retrying
            if attempt < max_retries:
                 await asyncio.sleep(2)

        # If all retries failed, return an empty structure
        logger.error(f"All retries failed for keyword '{keyword}'. Returning empty SERP data.")
        return self._create_empty_serp_data()


    def _extract_json_from_agent_result(self, result):
        """
        Extracts the JSON object from the agent's result.
        Prioritizes extracting from ActionResult objects, especially the final one with is_done=True.
        Falls back to string parsing the entire result if needed.

        Args:
            result: The raw result object from the agent (likely AgentHistoryList).

        Returns:
            dict: The parsed JSON object, or an empty dict if parsing fails or no JSON found.
        """
        if not result:
            logger.debug("_extract_json_from_agent_result received empty result.")
            return {}
        
        # 1. First priority: Extract from action_results() method if available
        if AgentHistoryList is not None and isinstance(result, AgentHistoryList):
            logger.debug(f"Result is AgentHistoryList. Attempting to extract using action_results() method.")
            
            try:
                # Try to use action_results() method if available
                action_results = result.action_results()
                if isinstance(action_results, list) and action_results:
                    logger.debug(f"Found {len(action_results)} action results using action_results() method")
                    
                    # First pass: Look specifically for ActionResults with is_done=True
                    for action_result in action_results:
                        if getattr(action_result, 'is_done', False) and getattr(action_result, 'success', False):
                            logger.debug("Found final successful ActionResult with is_done=True")
                            if hasattr(action_result, 'extracted_content') and action_result.extracted_content:
                                content = action_result.extracted_content
                                if isinstance(content, str):
                                    # Clean and try parsing the content
                                    content_str = content.strip()
                                    try:
                                        data = json.loads(content_str)
                                        logger.debug("Successfully parsed JSON from is_done=True ActionResult")
                                        return data
                                    except json.JSONDecodeError as e:
                                        logger.debug(f"Failed to parse JSON from final ActionResult: {e}")
                                elif isinstance(content, dict):
                                    # Already a dict, no need to parse
                                    logger.debug("Final ActionResult contained a dict, returning directly")
                                    return content
                    
                    # Second pass: Check all other ActionResults in reverse order
                    for action_result in reversed(action_results):
                        if hasattr(action_result, 'extracted_content') and action_result.extracted_content:
                            content = action_result.extracted_content
                            if isinstance(content, str):
                                content_str = content.strip()
                                if not content_str:
                                    continue
                                    
                                # Try direct JSON parsing
                                try:
                                    data = json.loads(content_str)
                                    logger.debug("Parsed JSON from ActionResult extracted_content")
                                    # Basic validation
                                    if isinstance(data, dict) and ("results" in data or "paa_questions" in data or "featured_snippet" in data):
                                        return data
                                except json.JSONDecodeError:
                                    # Try finding JSON within markdown blocks
                                    json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', content_str)
                                    if json_block_match:
                                        json_str = json_block_match.group(1).strip()
                                        try:
                                            data = json.loads(json_str)
                                            logger.debug("Parsed JSON from markdown block in ActionResult")
                                            # Basic validation
                                            if isinstance(data, dict) and ("results" in data or "paa_questions" in data or "featured_snippet" in data):
                                                return data
                                        except json.JSONDecodeError:
                                            pass
                            elif isinstance(content, dict):
                                # Already a dict, return directly
                                logger.debug("ActionResult contained a dict, returning directly")
                                # Basic validation
                                if "results" in content or "paa_questions" in content or "featured_snippet" in content:
                                    return content
            except AttributeError as e:
                logger.debug(f"Could not use action_results() method: {e}. Using fallback methods.")
                
            # Try using model_actions() as an alternative
            try:
                model_actions = result.model_actions()
                if isinstance(model_actions, list) and model_actions:
                    logger.debug(f"Found {len(model_actions)} model actions using model_actions() method")
                    for action in reversed(model_actions):
                        if isinstance(action, dict) and 'done' in action and isinstance(action['done'], dict):
                            done_action = action['done']
                            if 'text' in done_action and isinstance(done_action['text'], str):
                                try:
                                    data = json.loads(done_action['text'])
                                    logger.debug("Successfully parsed JSON from model_actions() method")
                                    return data
                                except json.JSONDecodeError:
                                    pass
            except AttributeError as e:
                logger.debug(f"Could not use model_actions() method: {e}. Using fallback methods.")

        # 2. Fallback: Try parsing from string representation as a last resort
        logger.debug("No valid JSON found in results. Falling back to parsing string representation of entire result object.")
        result_str = str(result)

        # Look for JSON within ```json ... ``` markdown
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', result_str)
        if json_block_match:
            json_str = json_block_match.group(1).strip()
            try:
                data = json.loads(json_str)
                logger.debug("Extracted JSON from ```json``` block in full string representation.")
                # Basic validation
                if isinstance(data, dict) and ("results" in data or "paa_questions" in data or "featured_snippet" in data):
                    return data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from ```json``` block in full string representation: {e}")
                # Fall through to next extraction methods

        # Look for JSON in the standard "📄 Result: {...}" pattern
        json_result_match = re.search(r'📄 Result:\s*(\{[\s\S]*?\})', result_str)
        if json_result_match:
            json_str = json_result_match.group(1).strip()
            try:
                data = json.loads(json_str)
                logger.debug("Extracted JSON from 📄 Result: pattern in full string representation.")
                # Basic validation
                if isinstance(data, dict) and ("results" in data or "paa_questions" in data or "featured_snippet" in data):
                    return data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from 📄 Result: pattern in full string representation: {e}")
                # Fall through to next extraction methods

        # Try finding any valid JSON object in the string as a last resort
        json_pattern_cautious = r'(\{.+?\})'
        potential_json_matches = re.findall(json_pattern_cautious, result_str, re.DOTALL)
        potential_json_matches.sort(key=len, reverse=True)

        for json_candidate in potential_json_matches:
            try:
                json_str = json_candidate.strip()
                data = json.loads(json_str)
                # Basic validation
                if isinstance(data, dict) and ("results" in data or "paa_questions" in data or "featured_snippet" in data):
                    logger.debug("Extracted JSON from general regex match in full string representation.")
                    return data
            except json.JSONDecodeError:
                continue # Try next match

        # If no valid JSON object was found after all attempts
        logger.warning("Could not extract a valid JSON object from agent result after all attempts.")
        return {} # Return empty dict if no valid JSON is found


    def _create_empty_serp_data(self):
        """Create empty SERP data structure for error cases"""
        return {
            "results": [],
            "featured_snippet": None,
            "paa_questions": [],
            "related_searches": [],
            "features": []
        }


    def _clean_url(self, url):
        """
        Clean and normalize a URL

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
