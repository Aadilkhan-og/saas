# Filename: src/modules/content_analyzer.py
import os
import re
import json
import logging
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI

# Removed GeminiClient import; using ChatGoogleGenerativeAI directly
# Keep browser_use for content analysis, as it requires rendering/interaction
try:
    from browser_use import Agent, AgentHistoryList, ActionResult # Import necessary classes
except ImportError:
    Agent = None
    AgentHistoryList = None
    ActionResult = None
    logging.error("browser_use not installed. Content analysis will not work.")

# Import textstat for readability calculation
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    textstat = None
    TEXTSTAT_AVAILABLE = False
    logging.warning("textstat library not installed. Readability scores will not be calculated.")


# Create module logger
logger = logging.getLogger("keyword_research.content_analyzer")

class ContentAnalyzer:
    """
    Analyzes competitor content from top-ranking pages using browser-use
    """

    def __init__(self):
        try:
            self.api_key = os.getenv("GOOGLE_API_KEY")
            self.model = "gemini-2.0-flash"
            if Agent:
                self.llm = ChatGoogleGenerativeAI(
                    model=self.model,
                    google_api_key=self.api_key,
                    temperature=0.7
                )
                logger.info("ChatGoogleGenerativeAI (Gemini) initialized for ContentAnalyzer")
                self.agent_available = True
            else:
                self.llm = None
                self.agent_available = False
                logger.error("browser_use not installed. Content analysis agent not available.")
        except Exception as e:
            logger.error(f"Error initializing ChatGoogleGenerativeAI in ContentAnalyzer: {str(e)}")
            self.llm = None
            self.agent_available = False
            self.agent_available = False


    async def analyze_content(self, urls, max_urls=5):
        """
        Analyze content from the provided URLs

        Args:
            urls (list): List of URLs to analyze
            max_urls (int): Maximum number of URLs to analyze

        Returns:
            dict: Analysis of competitor content
        """
        logger.info(f"Analyzing content from {len(urls)} URLs (max {max_urls})")

        # Initialize results dictionary
        results = {
            "analyzed_urls": [],
            "content_analysis": {}, # Detailed analysis per URL
            "common_themes": [], # Aggregated themes
            "content_types": {}, # Aggregated types count
            "heading_structure": {}, # Detailed headings (per URL)
            "content_length": {}, # Detailed word count (per URL)
            "readability_scores": {}, # Detailed readability (per URL) - NEW
            "summary": { # Aggregated summary
                "avg_word_count": 0,
                "avg_readability_score": 0, # NEW
                "content_types": {},
                "most_common_content_type": "unknown",
                "media_types": {},
                "content_freshness": "unknown", # Still heuristic/placeholder
                "common_themes_summary": [] # NEW - top themes list
            }
        }

        # Check if analysis agent is available
        if not self.agent_available:
            logger.error("Content analysis agent is not available. Skipping content analysis.")
            return results


        # Validate URLs before processing
        valid_urls = []
        for url in urls:
            # Clean up URLs and validate
            url = self._clean_url(url)
            if url and url.startswith(('http://', 'https://')):
                valid_urls.append(url)
            else:
                logger.warning(f"Invalid URL skipped during analysis: {url}")

        if not valid_urls:
            logger.warning("No valid URLs to analyze")
            return results

        # Limit the number of URLs to analyze
        urls_to_analyze = valid_urls[:min(max_urls, len(valid_urls))]
        logger.info(f"Processing {len(urls_to_analyze)} valid URLs: {urls_to_analyze}")

        # Initialize counters for summary data
        total_word_count = 0
        total_readability_score = 0 # NEW
        content_types_counter = {}
        media_types_counter = {}
        analyzed_count = 0
        all_themes = []

        # Process each URL
        tasks = [self._analyze_url_content_robust(url) for url in urls_to_analyze]
        analyzed_results = await asyncio.gather(*tasks)


        for url, content_data in zip(urls_to_analyze, analyzed_results):
             if content_data and "error" not in content_data:
                # Store results
                results["analyzed_urls"].append(url)
                results["content_analysis"][url] = content_data

                # Store content type - ensure it's not empty
                content_type = content_data.get("content_type", "unknown")
                if not content_type or content_type.lower() in ["null", "none", "unknown"]:
                    content_type = "unknown"
                results["content_types"][url] = content_type
                content_types_counter[content_type] = content_types_counter.get(content_type, 0) + 1

                # Store heading structure
                headings = content_data.get("headings", [])
                if isinstance(headings, list) and headings:
                    results["heading_structure"][url] = headings

                # Store content length - ensure it's a valid number
                word_count = content_data.get("word_count", 0)
                if word_count and isinstance(word_count, (int, float)) and word_count > 0:
                    results["content_length"][url] = word_count
                    total_word_count += word_count
                    # analyzed_count += 1 # Increment analyzed_count only if both word count and readability are successful below

                # Store readability score - NEW
                readability_score = content_data.get("readability_score")
                if readability_score is not None and isinstance(readability_score, (int, float)):
                     results["readability_scores"][url] = readability_score
                     total_readability_score += readability_score
                     analyzed_count += 1 # Increment only if both major metrics are captured
                     logger.info(f"Added word count ({word_count}) and readability ({readability_score}) for {url}")
                else:
                     logger.warning(f"Invalid or missing readability score for {url}: {readability_score}")
                     # If readability failed, don't count this URL towards the average
                     if word_count > 0: # Log that word count was found but not used for avg
                          logger.warning(f"Word count ({word_count}) found for {url} but not included in average due to missing readability.")


                # Track media types
                media_types = content_data.get("media_types", [])
                if isinstance(media_types, list):
                    for media_type in media_types:
                        if media_type and isinstance(media_type, str): # Skip empty or invalid values
                            media_types_counter[media_type.lower()] = media_types_counter.get(media_type.lower(), 0) + 1 # Use lowercase for consistency

                # Collect themes for later processing
                themes = content_data.get("key_themes", [])
                if isinstance(themes, list):
                    all_themes.extend([t for t in themes if t and isinstance(t, str)]) # Skip empty or invalid values

                logger.info(f"Successfully processed analysis data for URL: {url}")

             else:
                logger.warning(f"Failed to extract useful content analysis data from URL: {url}")


        # Process the collected data
        logger.info(f"Total word count: {total_word_count}, Total readability score: {total_readability_score}, Analyzed count for averages: {analyzed_count}")
        logger.info(f"Content types counter: {content_types_counter}")
        logger.info(f"Media types counter: {media_types_counter}")
        logger.info(f"All themes count: {len(all_themes)}")

        # Generate common themes from all collected themes
        if all_themes:
            results["common_themes"] = self._extract_common_themes_from_list(all_themes)
            logger.info(f"Extracted {len(results['common_themes'])} common themes across URLs")
            results["summary"]["common_themes_summary"] = results["common_themes"] # Add to summary


        # Calculate summary statistics
        if analyzed_count > 0:
            results["summary"]["avg_word_count"] = int(total_word_count / analyzed_count)
            results["summary"]["avg_readability_score"] = round(total_readability_score / analyzed_count, 1) # NEW - Avg readability
            logger.info(f"Average word count: {results['summary']['avg_word_count']}, Average readability: {results['summary']['avg_readability_score']}")
        else:
            logger.warning("No URLs with valid analysis data for averages.")
            results["summary"]["avg_word_count"] = 0
            results["summary"]["avg_readability_score"] = 0


        results["summary"]["content_types"] = content_types_counter

        # Determine most common content type
        if content_types_counter:
            # Sort content types by count descending and take the type
            sorted_content_types = sorted(content_types_counter.items(), key=lambda item: item[1], reverse=True)
            results["summary"]["most_common_content_type"] = sorted_content_types[0][0]
            logger.info(f"Most common content type: {results['summary']['most_common_content_type']}")
        else:
             results["summary"]["most_common_content_type"] = "unknown"


        results["summary"]["media_types"] = media_types_counter

        # Content freshness is still a placeholder/heuristic for now
        results["summary"]["content_freshness"] = "mixed" # Placeholder


        return results


    async def _analyze_url_content_robust(self, url):
        """
        Analyze content from a single URL with enhanced extraction and retries.

        Args:
            url (str): URL to analyze

        Returns:
            dict: Content analysis for the URL, including error key on failure
        """
        if not self.agent_available:
             logger.error("Content analysis agent not available, cannot analyze URL content.")
             return self._create_default_content_data(url, error="Agent not available")

        # Create a detailed task for the agent
        task = f"""
        Visit the URL '{url}' and carefully analyze the content to extract the following information.
        Focus on the main body content of the page, typical of an article, blog post, or product/service page.

        1.  **Page Title:** The exact title from the HTML <title> tag.
        2.  **Main Heading (H1):** The text content of the primary H1 tag on the page.
        3.  **Subheadings (H2 and H3):** A list of the text content of all H2 and H3 tags in order of their appearance.
        4.  **Main Text Content:** Extract the primary narrative or informational text content of the page's main body. Exclude headers, footers, sidebars, navigation, comments, and repetitive boilerplate text. Aim to capture the core written content.
        5.  **Word Count:** An approximate word count of the **Main Text Content** you extracted in step 4.
        6.  **Content Type:** Categorize the main type of content (e.g., "Blog Post", "Guide", "Product Page", "Landing Page", "Review", "Comparison", "Category Page", "Informational Article"). Be specific based on the page's primary purpose.
        7.  **Key Themes and Topics:** A list of the main themes, topics, or subjects discussed in the main body content. Focus on the core concepts.
        8.  **Media Types Present:** A list of significant media types found in the main content area (e.g., "Images", "Videos", "Infographics", "Interactive Elements", "Audio", "Text Only"). List "Text Only" if no significant media is present.
        9.  **Publication or Last Updated Date:** Find the publication date or last updated date if clearly visible on the page. Return null if not found.
        10. **Main Points/Thesis:** Briefly summarize the core message or main points the content is trying to convey (2-3 sentences).
        11. **Explicit Calls to Action (CTAs):** Identify any clear calls to action within the main content (e.g., "Buy Now", "Sign Up", "Download Guide", "Contact Us"). List the text of the CTA. Return an empty list if none are found.

        Return the results EXCLUSIVELY as a single, valid JSON object. Ensure all property names and string values within the JSON are correctly formatted and escaped. The JSON structure must be EXACTLY:
        {{
            "title": "Page Title",
            "h1": "Main Heading",
            "headings": ["Subheading 1", "Subheading 2", ...],
            "main_text_content": "The extracted main text content...", # NEW field
            "word_count": 1500, # Count of main_text_content
            "content_type": "Blog Post",
            "key_themes": ["Theme 1", "Theme 2", ...],
            "media_types": ["Images", "Videos", ...],
            "publication_date": "YYYY-MM-DD" or "Month Day, Year" or null,
            "main_points_summary": "Summary of main points...", # NEW field
            "calls_to_action": ["CTA Text 1", "CTA Text 2", ...] # NEW field
        }}

        Do NOT include any other text, markdown formatting (like ```json```), or explanation outside the single JSON object. If a field is not found or applicable, return null (for single values) or an empty array (for lists). Ensure the JSON is perfectly formatted for direct parsing.
        """

        max_retries = 3 # Increased retries for potential agent issues
        for attempt in range(max_retries + 1):
            try:
                # Create a browser-use agent
                agent = Agent(
                    task=task,
                    llm=self.llm,
                    # browser_config={"headless": True} # Keep headless if desired
                )

                # Run the agent
                logger.info(f"Running content analysis agent for URL (attempt {attempt+1}/{max_retries+1}): {url}")
                # Use a timeout for the agent run itself
                agent_run_timeout = 60 # seconds per URL, adjust as needed
                try:
                    result = await asyncio.wait_for(agent.run(), timeout=agent_run_timeout)
                    logger.info(f"Content analysis attempt completed for URL: {url}")
                except asyncio.TimeoutError:
                    logger.warning(f"Content analysis agent timed out for URL: {url} on attempt {attempt+1}. Retrying...")
                    if attempt < max_retries:
                        await asyncio.sleep(5) # Wait longer before retrying after timeout
                        continue
                    else:
                        logger.error(f"Content analysis agent timed out after {max_retries} retries for URL: {url}.")
                        return self._create_default_content_data(url, error="Analysis timed out")


                # Parse the result using enhanced extraction logic
                content_data = self._extract_content_data(result, url)

                # Post-process the data and calculate readability
                processed_data = self._postprocess_content_data(content_data, url) # Pass URL for logging/defaults

                # Log the processed data (excluding potentially large text content)
                loggable_data = processed_data.copy()
                if "main_text_content" in loggable_data:
                    loggable_data["main_text_content"] = f"[{len(loggable_data['main_text_content'] or '')} chars]" # Log char count instead of text
                logger.debug(f"Processed content data for '{url}' (Attempt {attempt+1}): {json.dumps(loggable_data, indent=2)}")


                # Basic validation: Check if we got at least a title and some text content or word count
                if processed_data.get("title") and (processed_data.get("main_text_content") or processed_data.get("word_count", 0) > 0):
                     logger.info(f"Successfully extracted useful content data on attempt {attempt+1} for URL: {url}")
                     return processed_data
                else:
                    logger.warning(f"Agent result for '{url}' did not return sufficient data (missing title or content/word count) on attempt {attempt+1}/{max_retries+1}. Retrying...")


            except Exception as e:
                logger.error(f"Error running or processing agent for '{url}' (attempt {attempt+1}/{max_retries+1}): {str(e)}", exc_info=True)
                # Error occurred, retry

            # Wait a bit before retrying
            if attempt < max_retries:
                 await asyncio.sleep(2)

        # If all retries failed, return a default structure with an error flag
        logger.error(f"All retries failed for URL '{url}'. Returning empty content data.")
        return self._create_default_content_data(url, error="Analysis failed after retries")


    def _postprocess_content_data(self, content_data, url):
        """
        Post-process raw extracted content data, calculate readability, and ensure validity.

        Args:
            content_data (dict): Raw content data extracted by the agent.
            url (str): The URL analyzed (for context/default data).

        Returns:
            dict: Processed and cleaned content data with added readability score.
        """
        # Start with a default structure to ensure all keys exist
        processed = self._create_default_content_data(url)

        # Update with extracted data if available and valid
        if isinstance(content_data, dict):
            processed.update({
                "title": content_data.get("title", processed["title"]),
                "h1": content_data.get("h1", processed["h1"]),
                "headings": content_data.get("headings", processed["headings"]),
                "main_text_content": content_data.get("main_text_content", processed["main_text_content"]), # Keep the raw text
                "word_count": content_data.get("word_count", processed["word_count"]),
                "content_type": content_data.get("content_type", processed["content_type"]),
                "key_themes": content_data.get("key_themes", processed["key_themes"]),
                "media_types": content_data.get("media_types", processed["media_types"]),
                "publication_date": content_data.get("publication_date", processed["publication_date"]),
                "main_points_summary": content_data.get("main_points_summary", processed["main_points_summary"]), # NEW
                "calls_to_action": content_data.get("calls_to_action", processed["calls_to_action"]), # NEW
                # Carry over error flag if it exists in the raw data
                "error": content_data.get("error", processed.get("error"))
            })


        # --- Cleaning and Validation ---

        # Ensure word_count is a non-negative integer
        try:
            if processed["word_count"] is not None:
                if isinstance(processed["word_count"], str):
                    # Remove non-numeric characters and convert
                    numeric_part = re.sub(r'[^0-9]', '', str(processed["word_count"]))
                    processed["word_count"] = int(numeric_part) if numeric_part else 0
                else:
                    processed["word_count"] = int(processed["word_count"])
            else:
                processed["word_count"] = 0
        except (ValueError, TypeError):
            logger.warning(f"Could not convert word_count to int for {url}. Value: {processed['word_count']}")
            processed["word_count"] = 0
        processed["word_count"] = max(0, processed["word_count"]) # Ensure non-negative


        # Ensure content_type is a string and lowercase for consistency
        if not isinstance(processed["content_type"], str) or not processed["content_type"].strip():
            processed["content_type"] = "unknown"
        else:
             processed["content_type"] = processed["content_type"].strip().lower()
             # Basic mapping for variations
             if processed["content_type"] in ["blog article", "article", "post"]:
                  processed["content_type"] = "blog post"
             elif processed["content_type"] in ["product page", "product information"]:
                  processed["content_type"] = "product page"
             elif processed["content_type"] in ["landing page", "sales page"]:
                  processed["content_type"] = "landing page"
             elif processed["content_type"] in ["review article", "product review"]:
                  processed["content_type"] = "review"
             elif processed["content_type"] in ["comparison article", "vs page"]:
                  processed["content_type"] = "comparison"
             elif processed["content_type"] in ["guide", "tutorial", "how-to"]:
                  processed["content_type"] = "guide"
             elif processed["content_type"] in ["category page", "product listing page"]:
                  processed["content_type"] = "category page"


        # Ensure lists are lists of non-empty strings
        for field in ["headings", "key_themes", "media_types", "calls_to_action"]:
            if not isinstance(processed.get(field), list):
                processed[field] = []
            else:
                 # Filter out any non-string or empty/whitespace-only values
                 processed[field] = [item.strip() for item in processed[field] if isinstance(item, str) and item.strip()]
                 # Convert media types to lowercase for consistency
                 if field == "media_types":
                      processed[field] = [item.lower() for item in processed[field]]


        # Ensure main_text_content is a string
        if not isinstance(processed["main_text_content"], str) or not processed["main_text_content"].strip():
             processed["main_text_content"] = "" # Ensure empty string if no text or just whitespace


        # Ensure main_points_summary is a string
        if not isinstance(processed["main_points_summary"], str):
             processed["main_points_summary"] = ""


        # Calculate Readability Score - NEW
        processed["readability_score"] = None # Default to None
        if TEXTSTAT_AVAILABLE and processed["main_text_content"]:
            try:
                # Using Flesch-Kincaid Grade Level - adjust if a different score is preferred
                # This score estimates the U.S. school grade level required to understand the text.
                fk_score = textstat.flesch_kincaid_grade(processed["main_text_content"])
                # textstat can sometimes return negative values for very simple text, cap at 0 or a small number
                processed["readability_score"] = max(0.0, round(fk_score, 1))
                logger.debug(f"Calculated readability score for {url}: {processed['readability_score']}")
            except Exception as e:
                logger.warning(f"Could not calculate readability score for {url}: {str(e)}")
                processed["readability_score"] = None # Explicitly set to None on error
        elif processed["main_text_content"]:
             logger.warning(f"textstat not available, could not calculate readability for {url}.")


        return processed


    def _extract_content_data(self, result, url):
        """
        Extract content data from agent result using multiple approaches.
        Enhanced to handle the new fields.

        Args:
            result: Agent result object (likely AgentHistoryList).
            url: URL being analyzed.

        Returns:
            dict: Extracted content data dictionary, or empty dict if parsing fails.
        """
        if not result:
            logger.debug("_extract_content_data received empty result.")
            return {}

        # Prioritize extraction from ActionResult objects, especially the final one
        if AgentHistoryList is not None and isinstance(result, AgentHistoryList):
            logger.debug(f"Result is AgentHistoryList. Attempting to extract from action_results().")
            try:
                action_results = result.action_results()
                if isinstance(action_results, list) and action_results:
                    # Try the latest ActionResult first, especially 'done' action
                    for action_result in reversed(action_results):
                         if hasattr(action_result, 'extracted_content') and action_result.extracted_content:
                              content = action_result.extracted_content
                              if isinstance(content, dict):
                                  # Found a dict directly, likely the parsed JSON
                                  logger.debug("Extracted dict from ActionResult.extracted_content")
                                  # Basic validation
                                  if 'title' in content or 'word_count' in content or 'main_text_content' in content:
                                       return content
                              elif isinstance(content, str):
                                  content_str = content.strip()
                                  if content_str:
                                       # Try parsing as JSON directly
                                       try:
                                            data = json.loads(content_str)
                                            logger.debug("Parsed JSON from ActionResult.extracted_content string")
                                            if 'title' in data or 'word_count' in data or 'main_text_content' in data:
                                                 return data
                                       except json.JSONDecodeError:
                                            # Try finding JSON within markdown block
                                            json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', content_str)
                                            if json_block_match:
                                                json_str = json_block_match.group(1).strip()
                                                try:
                                                    data = json.loads(json_str)
                                                    logger.debug("Parsed JSON from markdown block in ActionResult")
                                                    if 'title' in data or 'word_count' in data or 'main_text_content' in data:
                                                         return data
                                                except json.JSONDecodeError:
                                                    pass # Continue searching in other results

            except AttributeError as e:
                logger.debug(f"Could not use action_results() method or extracted_content: {e}. Using fallback methods.")
            except Exception as e:
                 logger.warning(f"Error during ActionResult extraction: {e}. Falling back.")


        # Fallback: Try parsing from string representation if ActionResult extraction failed
        logger.debug("Falling back to parsing string representation of entire result object.")
        result_str = str(result) # Get the string representation of the result object

        # Look for JSON within ```json ... ``` markdown block
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', result_str, re.DOTALL)
        if json_block_match:
            json_str = json_block_match.group(1).strip()
            try:
                data = json.loads(json_str)
                logger.debug("Extracted JSON from ```json``` block in full string representation.")
                # Basic validation for expected keys including new ones
                if isinstance(data, dict) and ('title' in data or 'word_count' in data or 'main_text_content' in data):
                    return data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from ```json``` block in full string representation: {e}")
                # Continue to next extraction methods

        # Look for JSON in the standard "塘 Result: {...}" pattern (common from browser-use)
        # This pattern might contain the JSON directly after the marker
        json_result_match = re.search(r'塘 Result:\s*(\{[\s\S]*?\})', result_str, re.DOTALL)
        if json_result_match:
            json_str = json_result_match.group(1).strip()
            try:
                data = json.loads(json_str)
                logger.debug("Extracted JSON from 塘 Result: pattern.")
                # Basic validation
                if isinstance(data, dict) and ('title' in data or 'word_count' in data or 'main_text_content' in data):
                    return data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from 塘 Result: pattern: {e}")
                # Fall through to next extraction methods


        # Try finding any valid JSON object in the string as a last resort, prioritizing longer ones
        # Use a cautious pattern to avoid matching partial structures
        json_pattern_cautious = r'(\{.*?)\}' # Non-greedy match from { to }
        potential_json_matches = re.findall(json_pattern_cautious, result_str, re.DOTALL)

        # Attempt to "fix" incomplete JSON objects by adding a closing brace if it seems truncated
        potential_json_matches = [match + '}' for match in potential_json_matches if not match.strip().endswith('}')] + [match for match in potential_json_matches if match.strip().endswith('}')]
        potential_json_matches.sort(key=len, reverse=True) # Prioritize longer potential JSON

        for json_candidate in potential_json_matches:
            try:
                json_str = json_candidate.strip()
                data = json.loads(json_str)
                # Basic validation: check for at least one core content field
                if isinstance(data, dict) and ('title' in data or 'word_count' in data or 'main_text_content' in data or 'key_themes' in data or 'headings' in data):
                    logger.debug("Extracted JSON from general regex match.")
                    return data
            except json.JSONDecodeError:
                continue # Try next match

        # If no valid JSON object was found after all attempts
        logger.warning(f"Could not extract a valid JSON object from agent result for URL '{url}' after all attempts.")
        return {} # Return empty dict if no valid JSON is found


    def _create_default_content_data(self, url, error=None):
        """
        Create default content data structure when analysis fails. Includes new fields.

        Args:
            url (str): URL that was analyzed.
            error (str, optional): An error message if analysis failed.

        Returns:
            dict: Default content data.
        """
        default_data = {
            "title": f"Analysis Failed: {url}",
            "h1": None,
            "headings": [],
            "main_text_content": "",
            "word_count": 0,
            "content_type": "unknown",
            "key_themes": [],
            "media_types": [], # Default to empty list now, not ["text"]
            "publication_date": None,
            "main_points_summary": None, # Default to None
            "calls_to_action": [],
            "readability_score": None, # Default readability to None
        }
        if error:
            default_data["error"] = error
        return default_data


    def _extract_common_themes_from_list(self, themes):
        """
        Extract common themes from a list of themes - Keep existing logic.

        Args:
            themes (list): List of themes (strings).

        Returns:
            list: Top common themes.
        """
        # Count frequency of each theme
        theme_frequency = {}
        for theme in themes:
            if theme: # Ensure theme is not empty
                theme_frequency[theme] = theme_frequency.get(theme, 0) + 1

        # Sort themes by frequency
        sorted_themes = sorted(theme_frequency.items(), key=lambda x: x[1], reverse=True)

        # Return the top themes (e.g., top 10 or adjust number)
        common_themes = [theme for theme, _ in sorted_themes[:10]]

        return common_themes

    # Method _extract_common_themes is no longer used directly in analyze_content,
    # the logic is integrated into the main loop. Keeping it might be useful
    # if you wanted a separate step later.
    # async def _extract_common_themes(self, content_analysis):
    #     """
    #     Extract common themes across all analyzed content
    #     """
    #     all_themes = []
    #     for url, data in content_analysis.items():
    #         all_themes.extend(data.get("key_themes", []))
    #     return self._extract_common_themes_from_list(all_themes)


    def _clean_url(self, url):
        """
        Clean and normalize URL - Keep existing logic

        Args:
            url (str): URL to clean

        Returns:
            str: Cleaned URL
        """
        # Remove quotes and extra characters
        if not url:
            return ""

        url = str(url)
        # Handle common browser-use artifact seen in logs "https:" instead of "https://"
        url = url.replace('"https":', 'https://').replace('"http":', 'http://')
        url = url.replace('"', '') # Remove any leftover quotes


        # Ensure URL has proper scheme
        if url and not url.startswith(('http://', 'https://')):
            if url and not url.isspace():
                 # Check if it looks like a domain/path, assume https
                if '.' in url: # Basic check for likely web address
                     url = 'https://' + url
                else:
                     # Looks like malformed data, return empty
                    logger.warning(f"URL '{url}' does not start with http/https and isn't a clear domain. Skipping.")
                    return ""
            else:
                 # Empty or whitespace after cleaning
                 return ""


        # Basic validation that it looks somewhat like a URL after cleaning
        if len(url) < 5 or '.' not in url:
            logger.warning(f"Cleaned URL '{url}' looks invalid. Skipping.")
            return ""


        return url.strip()