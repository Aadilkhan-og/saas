import os
import re
import json
import logging
import asyncio
from browser_use import Agent

# Create module logger
logger = logging.getLogger("keyword_research.content_analyzer")

class ContentAnalyzer:
    """
    Analyzes competitor content from top-ranking pages
    """
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        try:
            from modules.gemini_client import GeminiClient
            gemini_client = GeminiClient(api_key=self.google_api_key, model="gemini-2.0-flash")
            self.llm = gemini_client.get_langchain_llm()
            logger.info("Gemini initialized for ContentAnalyzer")
        except Exception as e:
            logger.error(f"Error initializing Gemini: {str(e)}")
            self.llm = None
    
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
            "content_analysis": {},
            "common_themes": [],
            "content_types": {},
            "heading_structure": {},
            "content_length": {},
            "summary": {
                "avg_word_count": 0,
                "content_types": {},
                "most_common_content_type": "unknown",
                "media_types": {},
                "content_freshness": "unknown"
            }
        }
        
        # Check if there are any URLs to analyze
        if not urls:
            logger.warning("No URLs to analyze")
            return results
            
        # Validate URLs before processing
        valid_urls = []
        for url in urls:
            # Clean up URLs and validate
            url = self._clean_url(url)
            if url and url.startswith(('http://', 'https://')):
                valid_urls.append(url)
            else:
                logger.warning(f"Invalid URL skipped: {url}")
        
        if not valid_urls:
            logger.warning("No valid URLs to analyze")
            return results
            
        # Limit the number of URLs to analyze
        urls_to_analyze = valid_urls[:min(max_urls, len(valid_urls))]
        logger.info(f"Processing {len(urls_to_analyze)} valid URLs: {urls_to_analyze}")
        
        # Initialize counters for summary data
        total_word_count = 0
        content_types_counter = {}
        media_types_counter = {}
        analyzed_count = 0
        all_themes = []
        
        # Process each URL
        for url in urls_to_analyze:
            try:
                # Analyze content for this URL
                content_data = await self._analyze_url_content(url)
                
                # Print content data for debugging
                logger.debug(f"Content data for {url}: {json.dumps(content_data, indent=2)}")
                
                # Check if we have valid content data that's not just the error template
                if content_data and "error" not in content_data:
                    # Store results
                    results["analyzed_urls"].append(url)
                    results["content_analysis"][url] = content_data
                    
                    # Store content type - ensure it's not empty
                    content_type = content_data.get("content_type", "unknown")
                    if not content_type or content_type.lower() == "null" or content_type.lower() == "none":
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
                        analyzed_count += 1
                        logger.info(f"Added word count for {url}: {word_count}")
                    else:
                        logger.warning(f"Invalid word count for {url}: {word_count}")
                    
                    # Track media types
                    media_types = content_data.get("media_types", [])
                    if isinstance(media_types, list):
                        for media_type in media_types:
                            if media_type:  # Skip empty values
                                media_types_counter[media_type] = media_types_counter.get(media_type, 0) + 1
                    
                    # Collect themes for later processing
                    themes = content_data.get("key_themes", [])
                    if isinstance(themes, list):
                        all_themes.extend([t for t in themes if t])  # Skip empty values
                    
                    logger.info(f"Successfully analyzed content for URL: {url}")
                else:
                    logger.warning(f"Failed to extract useful content from URL: {url}")
            
            except Exception as e:
                logger.error(f"Failed to analyze content for URL '{url}': {str(e)}", exc_info=True)
        
        # Process the collected data
        logger.info(f"Total word count: {total_word_count}, Analyzed count: {analyzed_count}")
        logger.info(f"Content types: {content_types_counter}")
        logger.info(f"Media types: {media_types_counter}")
        logger.info(f"All themes count: {len(all_themes)}")
        
        # Generate common themes from all collected themes
        if all_themes:
            results["common_themes"] = self._extract_common_themes_from_list(all_themes)
            logger.info(f"Extracted {len(results['common_themes'])} common themes")
        
        # Calculate summary statistics
        if analyzed_count > 0:
            results["summary"]["avg_word_count"] = int(total_word_count / analyzed_count)
            logger.info(f"Average word count: {results['summary']['avg_word_count']}")
        else:
            logger.warning("No analyzed URLs with valid word counts")
        
        results["summary"]["content_types"] = content_types_counter
        
        # Determine most common content type
        if content_types_counter:
            results["summary"]["most_common_content_type"] = max(content_types_counter.items(), key=lambda x: x[1])[0]
            logger.info(f"Most common content type: {results['summary']['most_common_content_type']}")
        
        results["summary"]["media_types"] = media_types_counter
        
        return results
    
    async def _analyze_url_content(self, url):
        """
        Analyze content from a single URL
        
        Args:
            url (str): URL to analyze
            
        Returns:
            dict: Content analysis for the URL
        """
        # Create a task for the agent
        task = f"""
        Visit the URL '{url}' and analyze the content. Extract the following information:
        
        1. Page title
        2. Main heading (H1)
        3. Subheadings (H2 and H3 tags)
        4. Word count (approximate)
        5. Content type (article, product page, landing page, etc.)
        6. Key themes and topics covered
        7. Media types present (text only, images, videos, interactive elements)
        8. Publication or last updated date (if available)
        
        Return the results in JSON format with the following structure:
        {{
            "title": "Page Title",
            "h1": "Main Heading",
            "headings": ["Subheading 1", "Subheading 2", ...],
            "word_count": 1500,
            "content_type": "blog article",
            "key_themes": ["Theme 1", "Theme 2", ...],
            "media_types": ["text", "images", ...],
            "publication_date": "2023-05-15" (or null if not found)
        }}
        
        Only return the JSON object, nothing else.
        """
        
        # Add retries for reliability
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Create and run the agent with headless mode to avoid emoji rendering issues
                if not self.llm:
                    raise ValueError("LLM not initialized")
                
                agent = Agent(
                    task=task,
                    llm=self.llm,
                )
                
                # Run the agent
                logger.info(f"Running content analysis agent for URL (attempt {attempt+1}/{max_retries+1}): {url}")
                result = await agent.run()
                logger.info(f"Content analysis completed for URL: {url}")
                
                # Parse the result using various methods
                content_data = self._extract_content_data(result, url)
                
                # Log the extracted data
                logger.info(f"Extracted content data for {url}: word_count={content_data.get('word_count')}, content_type={content_data.get('content_type')}")
                
                # Post-process the data to ensure validity
                processed_data = self._postprocess_content_data(content_data)
                
                # Only return if we have at least some meaningful data
                if processed_data.get("title") or processed_data.get("word_count"):
                    return processed_data
                else:
                    logger.warning(f"Extracted data lacks critical fields (title or word_count), retry attempt {attempt+1}/{max_retries+1}")
                    if attempt == max_retries:
                        # Last attempt failed, return whatever we got
                        return processed_data
            
            except Exception as e:
                logger.error(f"Error analyzing URL '{url}' (attempt {attempt+1}/{max_retries+1}): {str(e)}")
                if attempt == max_retries:
                    return self._create_default_content_data(url)
                
                # Wait a bit before retrying
                await asyncio.sleep(1)
        
        # Should never reach here but just in case
        return self._create_default_content_data(url)
    
    def _postprocess_content_data(self, content_data):
        """
        Post-process content data to ensure validity
        
        Args:
            content_data (dict): Raw content data
            
        Returns:
            dict: Processed content data
        """
        # Create a copy to avoid modifying the original
        processed = content_data.copy() if content_data else self._create_default_content_data("unknown")
        
        # Ensure word_count is an integer
        if "word_count" in processed:
            try:
                # Convert word count to integer
                if processed["word_count"] is not None:
                    if isinstance(processed["word_count"], str):
                        # Remove any non-numeric characters
                        numeric_part = re.sub(r'[^0-9]', '', processed["word_count"])
                        if numeric_part:
                            processed["word_count"] = int(numeric_part)
                        else:
                            processed["word_count"] = 0
                    else:
                        processed["word_count"] = int(processed["word_count"])
                else:
                    processed["word_count"] = 0
            except (ValueError, TypeError):
                processed["word_count"] = 0
        else:
            processed["word_count"] = 0
        
        # Ensure content_type is a string
        if "content_type" not in processed or not processed["content_type"]:
            processed["content_type"] = "unknown"
        
        # Ensure headings is a list
        if "headings" not in processed or not isinstance(processed["headings"], list):
            processed["headings"] = []
        
        # Ensure key_themes is a list
        if "key_themes" not in processed or not isinstance(processed["key_themes"], list):
            processed["key_themes"] = []
        
        # Ensure media_types is a list
        if "media_types" not in processed or not isinstance(processed["media_types"], list):
            processed["media_types"] = ["text"]  # Default to text
        
        return processed
    
    def _extract_content_data(self, result, url):
        """
        Extract content data from agent result using multiple approaches
        
        Args:
            result: Agent result object
            url: URL being analyzed
            
        Returns:
            dict: Extracted content data
        """
        try:
            # Dump the result object structure for debugging
            logger.debug(f"Result object type: {type(result)}")
            if hasattr(result, '__dict__'):
                logger.debug(f"Result attributes: {result.__dict__.keys()}")
            
            # 1. Try to extract direct result format (preferred format from my testing)
            result_str = str(result)
            if "Result: {" in result_str:
                # Extract between "Result: " and the next occurrence of "Task completed" or end of string
                result_json_str = result_str.split("Result: ", 1)[1]
                if "Task completed" in result_json_str:
                    result_json_str = result_json_str.split("Task completed", 1)[0]
                else:
                    # Take everything up to the last closing brace
                    last_brace = result_json_str.rfind("}")
                    if last_brace > 0:
                        result_json_str = result_json_str[:last_brace+1]
                
                # Clean up any leading/trailing whitespace
                result_json_str = result_json_str.strip()
                
                try:
                    # Try to parse the JSON
                    data = json.loads(result_json_str)
                    logger.info(f"Successfully extracted content data from direct result JSON: {url}")
                    return data
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse direct result JSON: {e}")
                    # Continue to other methods
            
            # 2. Try to extract from agent's direct result object
            if hasattr(result, '__dict__'):
                # First, check if the result itself has useful attributes
                if hasattr(result, 'title') and hasattr(result, 'word_count'):
                    return {
                        "title": getattr(result, 'title', ''),
                        "h1": getattr(result, 'h1', ''),
                        "headings": getattr(result, 'headings', []),
                        "word_count": getattr(result, 'word_count', 0),
                        "content_type": getattr(result, 'content_type', 'unknown'),
                        "key_themes": getattr(result, 'key_themes', []),
                        "media_types": getattr(result, 'media_types', []),
                        "publication_date": getattr(result, 'publication_date', None)
                    }
                
                # 3. Try to access agent's all_results for the done action
                all_results = getattr(result, 'all_results', None)
                if all_results and isinstance(all_results, list):
                    for action_result in all_results:
                        if hasattr(action_result, 'is_done') and action_result.is_done:
                            if hasattr(action_result, 'extracted_content'):
                                content = action_result.extracted_content
                                if isinstance(content, dict):
                                    return content
                                elif isinstance(content, str):
                                    try:
                                        return json.loads(content)
                                    except json.JSONDecodeError:
                                        # Continue to next approach
                                        pass
                
                # 4. Try model_outputs for the 'done' action
                model_outputs = getattr(result, 'all_model_outputs', None)
                if model_outputs and isinstance(model_outputs, list):
                    for output in model_outputs:
                        if isinstance(output, dict) and 'done' in output:
                            done_data = output.get('done', {})
                            if isinstance(done_data, dict) and 'text' in done_data:
                                text = done_data['text']
                                try:
                                    return json.loads(text)
                                except json.JSONDecodeError:
                                    # Will continue to text extraction
                                    pass
            
            # 5. Look for JSON in the standard "📄 Result: {...}" pattern
            json_pattern = r'📄 Result:\s*(\{[\s\S]*?\})'
            json_match = re.search(json_pattern, result_str)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    # Continue to next approach
                    pass
            
            # 6. Try to find any JSON object in the text
            json_pattern = r'(\{[\s\S]*?\})'
            json_matches = re.findall(json_pattern, result_str)
            
            # Try each JSON match, starting with the longest (most complete)
            json_matches.sort(key=len, reverse=True)
            
            for json_str in json_matches:
                try:
                    parsed = json.loads(json_str)
                    # Verify this is our content data by checking for expected keys
                    if 'title' in parsed or 'word_count' in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
            
            # 7. Try direct text pattern matching as last resort
            data = self._extract_content_from_text(result_str)
            if data.get('title') or data.get('word_count'):
                return data
            
            # 8. Fallback to default
            logger.warning(f"Could not extract content data from agent result for URL '{url}'")
            return self._create_default_content_data(url)
            
        except Exception as e:
            logger.error(f"Error extracting content data for URL '{url}': {str(e)}")
            return self._create_default_content_data(url)
    
    def _extract_content_from_text(self, text):
        """
        Extract content data directly from text using patterns
        
        Args:
            text (str): Text to extract from
            
        Returns:
            dict: Extracted content data
        """
        result = {
            "title": "",
            "h1": "",
            "headings": [],
            "word_count": 0,
            "content_type": "unknown",
            "key_themes": [],
            "media_types": [],
            "publication_date": None
        }
        
        try:
            # Extract title
            title_match = re.search(r'"title":\s*"([^"]+)"', text)
            if title_match:
                result["title"] = title_match.group(1)
            
            # Extract H1
            h1_match = re.search(r'"h1":\s*"([^"]+)"', text)
            if h1_match:
                result["h1"] = h1_match.group(1)
            
            # Extract word count
            word_count_match = re.search(r'"word_count":\s*(\d+)', text)
            if word_count_match:
                result["word_count"] = int(word_count_match.group(1))
            
            # Extract content type
            content_type_match = re.search(r'"content_type":\s*"([^"]+)"', text)
            if content_type_match:
                result["content_type"] = content_type_match.group(1)
            
            # Extract headings
            headings_match = re.search(r'"headings":\s*\[(.*?)\]', text, re.DOTALL)
            if headings_match:
                headings_text = headings_match.group(1)
                headings = re.findall(r'"([^"]+)"', headings_text)
                result["headings"] = headings
            
            # Extract key themes
            themes_match = re.search(r'"key_themes":\s*\[(.*?)\]', text, re.DOTALL)
            if themes_match:
                themes_text = themes_match.group(1)
                themes = re.findall(r'"([^"]+)"', themes_text)
                result["key_themes"] = themes
            
            # Extract media types
            media_match = re.search(r'"media_types":\s*\[(.*?)\]', text, re.DOTALL)
            if media_match:
                media_text = media_match.group(1)
                media = re.findall(r'"([^"]+)"', media_text)
                result["media_types"] = media
            
            # Extract publication date
            date_match = re.search(r'"publication_date":\s*"([^"]+)"', text)
            if date_match:
                result["publication_date"] = date_match.group(1)
            
            return result
        except Exception as e:
            logger.error(f"Error extracting content from text: {str(e)}")
            return result
    
    def _create_default_content_data(self, url):
        """
        Create default content data when analysis fails
        
        Args:
            url (str): URL that was analyzed
            
        Returns:
            dict: Default content data
        """
        return {
            "title": f"Failed to extract from {url}",
            "h1": "",
            "headings": [],
            "word_count": 0,
            "content_type": "unknown",
            "key_themes": [],
            "media_types": ["text"],
            "publication_date": None,
            "error": "Content extraction failed"
        }
    
    def _extract_common_themes_from_list(self, themes):
        """
        Extract common themes from a list of themes
        
        Args:
            themes (list): List of themes
            
        Returns:
            list: Common themes
        """
        # Count frequency of each theme
        theme_frequency = {}
        for theme in themes:
            theme_frequency[theme] = theme_frequency.get(theme, 0) + 1
        
        # Sort themes by frequency
        sorted_themes = sorted(theme_frequency.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top themes
        common_themes = [theme for theme, _ in sorted_themes[:10]]
        
        return common_themes
    
    async def _extract_common_themes(self, content_analysis):
        """
        Extract common themes across all analyzed content
        
        Args:
            content_analysis (dict): Dictionary of content analysis by URL
            
        Returns:
            list: Common themes across all content
        """
        # Collect all key themes from all URLs
        all_themes = []
        for url, data in content_analysis.items():
            all_themes.extend(data.get("key_themes", []))
        
        return self._extract_common_themes_from_list(all_themes)
    
    def _clean_url(self, url):
        """
        Clean and normalize URL
        
        Args:
            url (str): URL to clean
            
        Returns:
            str: Cleaned URL
        """
        # Remove quotes and extra characters
        if not url:
            return ""
        
        url = str(url)
        url = url.replace('"https":', 'https:').replace('"', '')
        
        # Ensure URL has proper scheme
        if not url.startswith(('http://', 'https://')):
            if url and not url.isspace():
                url = 'https://' + url
        
        return url.strip()
