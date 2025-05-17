import os
import logging
from modules.gemini_client import GeminiClient

logger = logging.getLogger("keyword_research.keyword_processor")

class KeywordProcessor:
    """
    Processes keywords to classify intent and cluster related terms
    """
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        try:
            gemini_client = GeminiClient(api_key=self.google_api_key, model="gemini-2.0-flash")
            self.llm = gemini_client.get_langchain_llm()
            logger.info("Gemini initialized for KeywordProcessor")
        except Exception as e:
            logger.error(f"Error initializing Gemini: {str(e)}")
            self.llm = None
        
        # Define intent categories
        self.intent_categories = {
            "informational": ["what", "how", "why", "who", "when", "where", "guide", "tutorial", "learn", "understand"],
            "navigational": ["official", "login", "website", "download", "app", "sign in", "account", "portal"],
            "commercial": ["best", "top", "review", "compare", "vs", "price", "cost", "worth", "alternative", "difference"],
            "transactional": ["buy", "purchase", "discount", "deal", "coupon", "shop", "order", "sale", "cheap", "affordable"]
        }
        
        # Define common intent phrases
        self.intent_phrases = {
            "informational": [
                "how to", "what is", "why does", "where to find", "when to", "who is",
                "guide to", "tutorial on", "learn about", "understand"
            ],
            "commercial": [
                "best", "top rated", "review of", "compare", "vs", "price of", 
                "cost of", "worth it", "alternative to", "difference between"
            ],
            "transactional": [
                "buy", "purchase", "get discount", "deals on", "coupon for", "shop for",
                "order online", "sale on", "cheap", "affordable"
            ]
        }
    
    def process_keywords(self, keywords, serp_data, competitor_data):
        """
        Process keywords to classify intent and cluster related terms
        
        Args:
            keywords (list): List of keywords to process
            serp_data (dict): SERP data collected for the keywords
            competitor_data (dict): Competitor content analysis
            
        Returns:
            dict: Processed keyword data
        """
        logger.info(f"Processing {len(keywords)} keywords")
        
        # Initialize results
        results = {
            "intent_classification": {},
            "clusters": [],
            "keyword_scores": {},
            "question_keywords": []
        }
        
        try:
            # Classify intent for each keyword
            for keyword in keywords:
                intent = self._classify_intent(keyword, serp_data)
                results["intent_classification"][keyword] = intent
                logger.debug(f"Classified keyword '{keyword}' as '{intent}'")
            
            # Generate keyword clusters
            results["clusters"] = self._cluster_keywords(keywords, results["intent_classification"])
            logger.info(f"Generated {len(results['clusters'])} keyword clusters")
            
            # Score keywords for difficulty and opportunity
            results["keyword_scores"] = self._score_keywords(keywords, serp_data, competitor_data)
            logger.info(f"Scored {len(results['keyword_scores'])} keywords")
            
            # Extract question-based keywords
            results["question_keywords"] = self._extract_questions(serp_data)
            logger.info(f"Extracted {len(results['question_keywords'])} question keywords")
            
            return results
        except Exception as e:
            logger.error(f"Error processing keywords: {str(e)}", exc_info=True)
            # Return partial results if available
            return results
    
    def _classify_intent(self, keyword, serp_data):
        """
        Classify the intent behind a keyword
        
        Args:
            keyword (str): Keyword to classify
            serp_data (dict): SERP data for classification context
            
        Returns:
            str: Intent classification (informational, navigational, commercial, transactional)
        """
        try:
            # Check for explicit intent signals in the keyword
            keyword_lower = keyword.lower()
            
            # First check common phrases (more accurate than individual word matching)
            for intent, phrases in self.intent_phrases.items():
                for phrase in phrases:
                    if phrase in keyword_lower:
                        return intent
            
            # Check each intent category for matching signals (individual words)
            for intent, signals in self.intent_categories.items():
                for signal in signals:
                    if signal in keyword_lower.split():
                        return intent
            
            # Check for question-based keywords (almost always informational)
            if keyword_lower.startswith(("what", "how", "why", "when", "where", "who", "which")):
                return "informational"
            
            # Check SERP features for intent signals
            if keyword in serp_data.get("features", {}):
                features = serp_data["features"].get(keyword, [])
                
                # Check for commercial/transactional signals
                if any(x in features for x in ["shopping_results", "price_results", "product_carousel"]):
                    return "transactional"
                
                # Check for navigational intent
                if "site_links" in features and len(serp_data.get("serp_data", {}).get(keyword, [])) > 0:
                    # Check if top result has the same domain as what's in the keyword
                    top_result = serp_data["serp_data"][keyword][0] if serp_data["serp_data"][keyword] else None
                    if top_result and "url" in top_result:
                        domain = self._extract_domain(top_result["url"])
                        # If keyword contains the domain, likely navigational
                        if domain in keyword_lower:
                            return "navigational"
                
                # Check for video intent
                if "video_carousel" in features:
                    return "informational"
            
            # Use contextual signals from SERP titles and descriptions
            if keyword in serp_data.get("serp_data", {}):
                serps = serp_data["serp_data"][keyword]
                
                # Count intent signals in titles and descriptions
                intent_counts = {
                    "informational": 0,
                    "navigational": 0,
                    "commercial": 0,
                    "transactional": 0
                }
                
                for result in serps[:5]:  # Check top 5 results
                    title = result.get("title", "").lower()
                    desc = result.get("description", "").lower()
                    combined = title + " " + desc
                    
                    # Count intent signals
                    for intent, signals in self.intent_categories.items():
                        for signal in signals:
                            if signal in combined.split():
                                intent_counts[intent] += 1
                
                # Find the intent with the highest count
                max_intent = max(intent_counts.items(), key=lambda x: x[1])
                if max_intent[1] > 0:
                    return max_intent[0]
            
            # Default to informational intent
            return "informational"
        except Exception as e:
            logger.error(f"Error classifying intent for keyword '{keyword}': {str(e)}")
            return "informational"  # Default to informational on error
    
    def _cluster_keywords(self, keywords, intent_classification):
        """
        Cluster keywords based on semantic similarity and intent
        
        Args:
            keywords (list): List of keywords to cluster
            intent_classification (dict): Intent classification for each keyword
            
        Returns:
            list: Clusters of related keywords
        """
        try:
            # Step 1: Group by intent
            intent_groups = {}
            for keyword, intent in intent_classification.items():
                if intent not in intent_groups:
                    intent_groups[intent] = []
                intent_groups[intent].append(keyword)
            
            # Step 2: For each intent group, cluster by topic similarity
            all_clusters = []
            
            for intent, intent_keywords in intent_groups.items():
                # Skip if there are no keywords for this intent
                if not intent_keywords:
                    continue
                
                # For each intent, create topic clusters
                topic_clusters = self._create_topic_clusters(intent_keywords)
                
                # Add intent information to each cluster
                for cluster in topic_clusters:
                    cluster["intent"] = intent
                    all_clusters.append(cluster)
            
            return all_clusters
        except Exception as e:
            logger.error(f"Error clustering keywords: {str(e)}")
            # Return simple clusters based on intent only as fallback
            fallback_clusters = []
            for intent, keywords in intent_classification.items():
                if keywords:
                    fallback_clusters.append({
                        "intent": intent,
                        "topic": intent,
                        "keywords": list(keywords)
                    })
            return fallback_clusters
    
    def _create_topic_clusters(self, keywords):
        """
        Create topic clusters from a list of keywords
        
        Args:
            keywords (list): List of keywords to cluster
            
        Returns:
            list: Topic clusters
        """
        try:
            # Simple clustering based on common words and phrases
            word_to_keywords = {}
            
            # Map significant words to keywords containing them
            for keyword in keywords:
                words = keyword.lower().split()
                # Filter out stop words
                significant_words = [w for w in words if len(w) > 3 and w not in [
                    "what", "when", "where", "which", "how", "why", "who", "that", "this", "then", "than",
                    "with", "from", "your", "have", "will", "should", "could", "would", "best", "most"
                ]]
                
                for word in significant_words:
                    if word not in word_to_keywords:
                        word_to_keywords[word] = set()
                    word_to_keywords[word].add(keyword)
            
            # Create clusters based on words that appear in multiple keywords
            clusters = []
            processed_keywords = set()
            
            # Sort words by frequency (number of keywords containing them)
            sorted_words = sorted(
                word_to_keywords.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )
            
            for word, word_keywords in sorted_words:
                # Skip if this word appears in only one keyword
                if len(word_keywords) < 2:
                    continue
                
                # Skip words that are already fully covered by existing clusters
                if all(kw in processed_keywords for kw in word_keywords):
                    continue
                
                # Create a new cluster
                new_keywords = [kw for kw in word_keywords if kw not in processed_keywords]
                if new_keywords:
                    clusters.append({
                        "topic": word,
                        "keywords": new_keywords
                    })
                    processed_keywords.update(new_keywords)
            
            # Add any remaining unclustered keywords as individual clusters
            for keyword in keywords:
                if keyword not in processed_keywords:
                    # Find the most significant word in the keyword
                    words = keyword.lower().split()
                    significant_words = [w for w in words if len(w) > 3]
                    topic = max(significant_words, key=len) if significant_words else keyword
                    
                    clusters.append({
                        "topic": topic,
                        "keywords": [keyword]
                    })
                    processed_keywords.add(keyword)
            
            return clusters
        except Exception as e:
            logger.error(f"Error creating topic clusters: {str(e)}")
            # Create a single cluster with all keywords as fallback
            return [{
                "topic": "general",
                "keywords": keywords
            }]
    
    def _score_keywords(self, keywords, serp_data, competitor_data):
        """
        Score keywords for difficulty and opportunity
        
        Args:
            keywords (list): List of keywords to score
            serp_data (dict): SERP data for the keywords
            competitor_data (dict): Competitor content analysis
            
        Returns:
            dict: Scores for each keyword
        """
        try:
            scores = {}
            
            # Define major domains that indicate higher competition
            major_domains = [
                "wikipedia.org", "amazon.com", "youtube.com", "facebook.com", 
                "instagram.com", "linkedin.com", "twitter.com", "reddit.com",
                "nytimes.com", "washingtonpost.com", "wsj.com", "forbes.com",
                "cnn.com", "bbc.com", "theguardian.com", "huffpost.com"
            ]
            
            for keyword in keywords:
                # Initialize scores
                difficulty = 50  # Medium difficulty by default
                opportunity = 50  # Medium opportunity by default
                
                # Classify the intent to help with scoring
                intent = "informational"
                if keyword in serp_data.get("intent_classification", {}):
                    intent = serp_data.get("intent_classification", {}).get(keyword, "informational")
                
                # Adjust difficulty based on SERP data if available
                if keyword in serp_data.get("serp_data", {}):
                    serps = serp_data["serp_data"][keyword]
                    
                    # Check domain authority (simplified for demo)
                    top_domains = []
                    for s in serps[:5]:  # Consider top 5 results
                        if "url" in s:
                            domain = self._extract_domain(s["url"])
                            top_domains.append(domain)
                    
                    major_domain_count = sum(1 for domain in top_domains if any(major in domain for major in major_domains))
                    
                    # Increase difficulty if major domains dominate the SERP
                    if major_domain_count >= 3:
                        difficulty += 30
                    elif major_domain_count >= 2:
                        difficulty += 20
                    elif major_domain_count == 1:
                        difficulty += 10
                    elif major_domain_count == 0:
                        difficulty -= 20
                    
                    # Adjust opportunity based on SERP features
                    if keyword in serp_data.get("features", {}):
                        features = serp_data["features"][keyword]
                        
                        # More features generally means more opportunity
                        opportunity += min(len(features) * 5, 20)
                        
                        # Featured snippets provide good opportunity
                        if "featured_snippet" in features:
                            opportunity += 15
                            
                        # Video carousels can provide opportunity
                        if "video_carousel" in features:
                            opportunity += 10
                            
                        # PAA questions provide opportunity
                        if keyword in serp_data.get("paa_questions", {}) and serp_data["paa_questions"][keyword]:
                            opportunity += min(len(serp_data["paa_questions"][keyword]) * 3, 15)
                
                # Adjust based on intent
                if intent == "informational":
                    # Informational keywords generally have more content opportunities
                    opportunity += 5
                elif intent == "commercial":
                    # Commercial keywords can be lucrative but competitive
                    difficulty += 10
                    opportunity += 15
                elif intent == "transactional":
                    # Transactional keywords are often highly competitive
                    difficulty += 20
                    opportunity += 20
                
                # Adjust based on keyword complexity
                word_count = len(keyword.split())
                if word_count >= 4:
                    # Long-tail keywords are less competitive
                    difficulty -= 15
                    # But may have less search volume
                    opportunity -= 5
                
                # Adjust based on competitor content
                if competitor_data.get("summary", {}).get("avg_word_count", 0) > 2000:
                    # Long-form content indicates higher competition
                    difficulty += 10
                
                # Ensure scores are within 0-100 range
                difficulty = max(0, min(100, difficulty))
                opportunity = max(0, min(100, opportunity))
                
                # Calculate combined score (favoring opportunity)
                combined_score = round((opportunity * 0.7) - (difficulty * 0.3) + 50)
                combined_score = max(0, min(100, combined_score))
                
                # Store scores
                scores[keyword] = {
                    "difficulty": round(difficulty),
                    "opportunity": round(opportunity),
                    "score": round(combined_score)
                }
            
            return scores
        except Exception as e:
            logger.error(f"Error scoring keywords: {str(e)}")
            # Return empty scores on error
            return {keyword: {"difficulty": 50, "opportunity": 50, "score": 50} for keyword in keywords}
    
    def _extract_questions(self, serp_data):
        """
        Extract question-based keywords from SERP data
        
        Args:
            serp_data (dict): SERP data containing PAA questions
            
        Returns:
            list: Question-based keywords
        """
        try:
            questions = []
            
            # Extract PAA questions from all keywords
            for keyword, paa_questions in serp_data.get("paa_questions", {}).items():
                questions.extend(paa_questions)
            
            # Also look for question formats in related searches
            for keyword, related_searches in serp_data.get("related_searches", {}).items():
                for related in related_searches:
                    if any(related.lower().startswith(q) for q in ["what", "how", "why", "when", "where", "who", "which"]):
                        questions.append(related)
            
            # Remove duplicates
            questions = list(dict.fromkeys(questions))
            
            return questions
        except Exception as e:
            logger.error(f"Error extracting questions: {str(e)}")
            return []
    
    def _extract_domain(self, url):
        """
        Extract domain from a URL
        
        Args:
            url (str): URL to extract domain from
            
        Returns:
            str: Domain name
        """
        try:
            from urllib.parse import urlparse
            
            # Clean up URL
            if '"https":' in url:
                url = url.replace('"https":', 'https:')
            url = url.replace('"', '')
            
            # Parse URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
        except Exception as e:
            logger.error(f"Error extracting domain from URL '{url}': {str(e)}")
            # Return original URL if parsing fails
            return url
