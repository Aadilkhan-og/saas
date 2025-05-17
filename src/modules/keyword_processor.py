# Filename: src/modules/keyword_processor.py
import os
import logging
import re
import numpy as np

from langchain_google_genai import ChatGoogleGenerativeAI

# Import libraries for NLP clustering
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    import torch # Needed for SentenceTransformer device management
    NLP_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    AgglomerativeClustering = None
    cosine_similarity = None
    torch = None
    NLP_AVAILABLE = False
    logging.error("NLP libraries (sentence-transformers, scikit-learn) not installed. Advanced clustering will not work.")


logger = logging.getLogger("keyword_research.keyword_processor")

class KeywordProcessor:
    """
    Processes keywords to classify intent, cluster related terms, and score them.
    Uses SerpAPI data for enhanced scoring and NLP for clustering.
    """

    def __init__(self):
        global NLP_AVAILABLE
        try:
            self.api_key = os.getenv("GOOGLE_API_KEY")
            self.model = "gemini-2.0-flash"
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=0.7
            )
            logger.info("ChatGoogleGenerativeAI (Gemini) initialized for KeywordProcessor")
        except Exception as e:
            logger.error(f"Error initializing ChatGoogleGenerativeAI in KeywordProcessor: {str(e)}")
            self.llm = None

        # Define intent categories (kept as is)
        self.intent_categories = {
            "informational": ["what", "how", "why", "who", "when", "where", "guide", "tutorial", "learn", "understand"],
            "navigational": ["official", "login", "website", "download", "app", "sign in", "account", "portal"],
            "commercial": ["best", "top", "review", "compare", "vs", "price", "cost", "worth", "alternative", "difference"],
            "transactional": ["buy", "purchase", "discount", "deal", "coupon", "shop", "order", "sale", "cheap", "affordable"]
        }

        # Define common intent phrases (kept as is)
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

        self.sentence_model = None
        if NLP_AVAILABLE:
            try:
                # Load a pre-trained sentence transformer model
                # You might choose a different model depending on performance/accuracy needs
                # 'all- MiniLM-L6-v2' is a good balance
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                logger.info(f"SentenceTransformer model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model: {str(e)}")
                self.sentence_model = None
                NLP_AVAILABLE = False # Disable NLP if model loading fails


    def process_keywords(self, keywords, serp_data, competitor_data):
        """
        Process keywords to classify intent, cluster related terms, and score them.

        Args:
            keywords (list): List of keywords to process
            serp_data (dict): SERP data collected for the keywords (includes SV, KD)
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
            # Use SerpAPI features and organic results snippets for better context
            for keyword in keywords:
                intent = self._classify_intent(
                    keyword,
                    serp_data.get("features", {}).get(keyword, []), # Pass features for keyword
                    serp_data.get("serp_data", {}).get(keyword, []) # Pass organic results for keyword
                    )
                results["intent_classification"][keyword] = intent
                logger.debug(f"Classified keyword '{keyword}' as '{intent}'")

            # Generate keyword clusters using NLP if available
            if NLP_AVAILABLE and self.sentence_model:
                 logger.info("Attempting NLP-based keyword clustering")
                 results["clusters"] = self._cluster_keywords_nlp(keywords, results["intent_classification"])
                 logger.info(f"Generated {len(results['clusters'])} NLP keyword clusters")
            else:
                 logger.warning("NLP not available, falling back to basic keyword clustering.")
                 results["clusters"] = self._cluster_keywords_basic(keywords, results["intent_classification"])
                 logger.info(f"Generated {len(results['clusters'])} basic keyword clusters")


            # Score keywords for difficulty and opportunity using SV/KD
            results["keyword_scores"] = self._score_keywords(
                keywords,
                serp_data.get("search_volume", {}), # Pass SV data
                serp_data.get("keyword_difficulty", {}), # Pass KD data
                serp_data # Pass full serp_data for other signals if needed
            )
            logger.info(f"Scored {len(results['keyword_scores'])} keywords")

            # Extract question-based keywords from PAA (using new serp_data structure)
            results["question_keywords"] = self._extract_questions(serp_data.get("paa_questions", {}))
            logger.info(f"Extracted {len(results['question_keywords'])} question keywords")

            return results
        except Exception as e:
            logger.error(f"Error processing keywords: {str(e)}", exc_info=True)
            # Return partial results if available
            return results

    def _classify_intent(self, keyword, serp_features, organic_results):
        """
        Classify the intent behind a keyword using keyword signals and SERP features/snippets.

        Args:
            keyword (str): Keyword to classify
            serp_features (list): List of detected SERP features for the keyword
            organic_results (list): List of organic search results (title, snippet, link)

        Returns:
            str: Intent classification (informational, navigational, commercial, transactional)
        """
        try:
            keyword_lower = keyword.lower()

            # 1. Check for explicit intent signals in the keyword (phrases first, then words)
            for intent, phrases in self.intent_phrases.items():
                for phrase in phrases:
                    if phrase in keyword_lower:
                        return intent

            for intent, signals in self.intent_categories.items():
                for signal in signals:
                    if signal in keyword_lower.split():
                        return intent

            # Check for question-based keywords (strong signal for informational)
            if keyword_lower.startswith(("what", "how", "why", "when", "where", "who", "which", "can i", "are", "is")):
                return "informational"

            # 2. Check SERP features for intent signals (using the list directly)
            if serp_features:
                # Check for commercial/transactional signals
                if any(x in serp_features for x in ["shopping_results", "product_carousel", "price_results"]):
                    return "transactional" # More specific than just commercial

                # Check for informational intent signals
                if any(x in serp_features for x in ["featured_snippet", "people_also_ask", "video_results"]):
                     # If informational features are present, but commercial/transactional aren't strong, lean informational
                     if not any(x in serp_features for x in ["shopping_results", "product_carousel", "price_results"]):
                          return "informational"
                     # If both are present, analyze results snippets
                     pass # Continue to snippet analysis

                # Check for navigational intent (e.g., brand name + "login" or "website")
                # This is harder without explicit brand data, but can check for site_links + brand in keyword
                if "site_links" in serp_features and len(organic_results) > 0:
                     top_result = organic_results[0]
                     if top_result and "link" in top_result:
                        domain = self._extract_domain(top_result["link"])
                        # Basic check: if keyword contains the domain name
                        if domain.replace("www.", "") in keyword_lower:
                            return "navigational"


            # 3. Use contextual signals from SERP titles and snippets
            if organic_results:
                intent_counts = {
                    "informational": 0,
                    "navigational": 0,
                    "commercial": 0,
                    "transactional": 0
                }

                for result in organic_results[:5]:  # Check top 5 results
                    title = result.get("title", "").lower()
                    snippet = result.get("snippet", "").lower()
                    combined = title + " " + snippet

                    # Count intent signals in snippets and titles
                    for intent, signals in self.intent_categories.items():
                        for signal in signals:
                            if signal in combined: # Use 'in combined' for phrases/words in text
                                intent_counts[intent] += 1

                # Find the intent with the highest count
                # Sort by count, then prioritize transactional > commercial > navigational > informational
                sorted_intents = sorted(
                    intent_counts.items(),
                    key=lambda x: (x[1], list(self.intent_categories.keys()).index(x[0])),
                    reverse=True # Higher count first, then reverse index (transactional=3, commercial=2...)
                    )

                if sorted_intents and sorted_intents[0][1] > 0:
                    return sorted_intents[0][0]

            # Default to informational intent if no strong signals found
            return "informational"
        except Exception as e:
            logger.error(f"Error classifying intent for keyword '{keyword}': {str(e)}")
            return "unknown" # Use "unknown" instead of defaulting to informational on error

    def _cluster_keywords_nlp(self, keywords, intent_classification):
        """
        Cluster keywords based on semantic similarity using NLP embeddings.

        Args:
            keywords (list): List of keywords to cluster
            intent_classification (dict): Intent classification for each keyword

        Returns:
            list: Clusters of related keywords with intent and topic
        """
        if not NLP_AVAILABLE or not self.sentence_model or not keywords:
            logger.warning("NLP not available or no keywords provided for clustering.")
            return self._cluster_keywords_basic(keywords, intent_classification) # Fallback to basic

        try:
            # Get embeddings for all keywords
            # Using encode with convert_to_numpy=True
            keyword_embeddings = self.sentence_model.encode(keywords, convert_to_numpy=True, show_progress_bar=False)

            # Calculate cosine similarity matrix
            # Use float32 to save memory if needed, depending on scale
            similarity_matrix = cosine_similarity(keyword_embeddings.astype(np.float32))

            # Apply Agglomerative Clustering
            # You can adjust the distance_threshold. 0.7 is a common starting point for cosine similarity.
            # Linkage can be 'ward', 'complete', 'average', 'single'. 'average' or 'complete' often work well for text.
            # n_clusters=None and distance_threshold requires scikit-learn >= 0.21
            # If you have an older version or want a fixed number of clusters, use n_clusters=N
            # If distance_threshold clustering is not yielding good results, try a fixed n_clusters
            clustering_model = AgglomerativeClustering(
                n_clusters=None, # Automatically determine number of clusters
                distance_threshold=0.3, # Cosine distance threshold (1 - cosine similarity)
                linkage='average' # Linkage method
            )

            # Fit the model and get cluster labels
            # Reshape similarity_matrix if affinity='precomputed' is used
            # If using affinity='cosine' with a similarity matrix, sklearn might need 1-similarity (distance)
            # Let's re-calculate as distance matrix for affinity='precomputed'
            distance_matrix = 1 - similarity_matrix
            clustering_model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.3,
                linkage='average'
            )
            cluster_labels = clustering_model.fit_predict(distance_matrix)


            # Organize keywords by cluster label
            clustered_keywords = {}
            for i, label in enumerate(cluster_labels):
                # Skip noise points if using a density-based method like DBSCAN/HDBSCAN which can return -1
                # Agglomerative usually assigns all points, but check if using others
                if label == -1:
                     continue
                if label not in clustered_keywords:
                    clustered_keywords[label] = []
                clustered_keywords[label].append(keywords[i])

            # Format results
            clusters = []
            for label, cluster_keywords_list in clustered_keywords.items():
                 # Determine the most frequent intent within the cluster
                 intent_counts = {}
                 for kw in cluster_keywords_list:
                      intent = intent_classification.get(kw, "unknown")
                      intent_counts[intent] = intent_counts.get(intent, 0) + 1
                 # Use the overall most common intent, or the most common in the cluster
                 # For simplicity, let's use the most common in the cluster
                 most_common_intent_in_cluster = max(intent_counts.items(), key=lambda item: item[1])[0] if intent_counts else "unknown"


                 # Attempt to find a representative topic word/phrase for the cluster
                 # This can be done by finding the keyword closest to the centroid of the cluster embeddings
                 # Or by finding common words/phrases in the keywords (fallback)
                 topic = f"Cluster {label}" # Default topic
                 if cluster_keywords_list:
                     try:
                          # Get embeddings for keywords in this cluster
                          cluster_indices = [keywords.index(kw) for kw in cluster_keywords_list]
                          cluster_embeddings = keyword_embeddings[cluster_indices]

                          # Calculate cluster centroid
                          centroid = np.mean(cluster_embeddings, axis=0)

                          # Find keyword closest to the centroid
                          # Calculate cosine similarity between centroid and each keyword in the cluster
                          centroid_similarity = cosine_similarity([centroid], cluster_embeddings)[0]
                          closest_keyword_index_in_cluster = np.argmax(centroid_similarity)
                          topic = cluster_keywords_list[closest_keyword_index_in_cluster]
                     except Exception as e:
                           logger.debug(f"Could not find centroid for cluster {label}: {str(e)}. Using basic topic finding.")
                           # Fallback to basic topic finding if centroid method fails
                           topic = self._find_basic_topic_for_cluster(cluster_keywords_list) or f"Cluster {label}"


                 clusters.append({
                     "topic": topic,
                     "intent": most_common_intent_in_cluster,
                     "keywords": cluster_keywords_list
                 })

            return clusters

        except Exception as e:
            logger.error(f"Error during NLP-based clustering: {str(e)}", exc_info=True)
            return self._cluster_keywords_basic(keywords, intent_classification) # Fallback on error


    def _cluster_keywords_basic(self, keywords, intent_classification):
        """
        Basic keyword clustering based on common words and intent (fallback method).

        Args:
            keywords (list): List of keywords to cluster
            intent_classification (dict): Intent classification for each keyword

        Returns:
            list: Topic clusters
        """
        logger.warning("Using basic keyword clustering.")
        try:
            # Step 1: Group by intent
            intent_groups = {}
            for keyword in keywords:
                intent = intent_classification.get(keyword, "unknown")
                if intent not in intent_groups:
                    intent_groups[intent] = []
                intent_groups[intent].append(keyword)

            # Step 2: For each intent group, cluster by topic similarity (simplified)
            all_clusters = []

            for intent, intent_keywords in intent_groups.items():
                if not intent_keywords:
                    continue

                # Simple clustering within intent group based on shared words
                clusters_in_intent = self._create_topic_clusters_basic(intent_keywords)

                # Add intent information to each cluster
                for cluster in clusters_in_intent:
                    cluster["intent"] = intent
                    all_clusters.append(cluster)

            return all_clusters
        except Exception as e:
            logger.error(f"Error during basic keyword clustering: {str(e)}")
            # Create a single cluster with all keywords as fallback
            return [{
                "topic": "general",
                "intent": "unknown",
                "keywords": list(keywords)
            }]


    def _create_topic_clusters_basic(self, keywords):
        """
        Create topic clusters from a list of keywords using basic word matching.

        Args:
            keywords (list): List of keywords to cluster

        Returns:
            list: Topic clusters
        """
        try:
            # Simple clustering based on common words and phrases
            word_to_keywords = {}

            # Define a slightly more extensive list of common English stop words
            stop_words = set([
                "a", "an", "the", "and", "or", "but", "about", "above", "after", "before", "behind",
                "below", "beneath", "beside", "between", "beyond", "by", "down", "during", "for",
                "from", "in", "into", "like", "near", "of", "off", "on", "onto", "out", "outside",
                "over", "past", "through", "to", "under", "up", "with", "without", "at", "by",
                "for", "from", "in", "of", "on", "to", "with", "vs", "vs.", "vs", "vs.", "best", "how", "what", "why", "when", "where", "who", "which", "guide", "tutorial", "review", "compare", "price", "cost", "buy", "shop", "online", "near", "me", "find", "get", "list", "top"
            ])


            # Map significant words to keywords containing them
            for keyword in keywords:
                # Remove punctuation and split into words
                words = re.findall(r'\b\w+\b', keyword.lower())
                # Filter out stop words and short words
                significant_words = [w for w in words if w not in stop_words and len(w) > 2] # Keep slightly shorter words


                for word in significant_words:
                    if word not in word_to_keywords:
                        word_to_keywords[word] = set()
                    word_to_keywords[word].add(keyword)

            # Create clusters based on words that appear in multiple keywords
            clusters = []
            processed_keywords = set()

            # Sort words by frequency (number of keywords containing them)
            # Consider words that appear in at least 2 keywords as potential cluster centers
            sorted_words = sorted(
                [(word, list(kws)) for word, kws in word_to_keywords.items() if len(kws) > 1],
                key=lambda x: len(x[1]),
                reverse=True
            )

            for word, word_keywords in sorted_words:
                # Create a new cluster if it contains keywords not yet processed
                new_keywords = [kw for kw in word_keywords if kw not in processed_keywords]
                if new_keywords:
                    clusters.append({
                        "topic": word, # Use the common word as the topic
                        "keywords": new_keywords
                    })
                    processed_keywords.update(new_keywords)

            # Add any remaining unclustered keywords as individual clusters
            for keyword in keywords:
                if keyword not in processed_keywords:
                    # Try to find a descriptive word for single keyword cluster
                    words = re.findall(r'\b\w+\b', keyword.lower())
                    significant_words = [w for w in words if w not in stop_words and len(w) > 2]
                    topic = significant_words[0] if significant_words else keyword # Use first significant word or the keyword itself

                    clusters.append({
                        "topic": topic,
                        "keywords": [keyword]
                    })
                    processed_keywords.add(keyword)

            return clusters
        except Exception as e:
            logger.error(f"Error creating basic topic clusters: {str(e)}")
            # Create a single cluster with all keywords as fallback
            return [{
                "topic": "general",
                "keywords": list(keywords) # Ensure it's a list
            }]

    def _find_basic_topic_for_cluster(self, keywords):
        """Helper to find a basic topic for a cluster by finding common words."""
        if not keywords:
            return None

        word_counts = {}
        stop_words = set([
            "a", "an", "the", "and", "or", "but", "about", "above", "after", "before", "behind",
            "below", "beneath", "beside", "between", "beyond", "by", "down", "during", "for",
            "from", "in", "into", "like", "near", "of", "off", "on", "onto", "out", "outside",
            "over", "past", "through", "to", "under", "up", "with", "without", "at", "by",
            "for", "from", "in", "of", "on", "to", "with", "vs", "vs.", "best", "how", "what", "why", "when", "where", "who", "which", "guide", "tutorial", "review", "compare", "price", "cost", "buy", "shop", "online", "near", "me", "find", "get", "list", "top"
        ])


        for keyword in keywords:
            words = re.findall(r'\b\w+\b', keyword.lower())
            significant_words = [w for w in words if w not in stop_words and len(w) > 2]
            for word in significant_words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Return the most frequent word
        if word_counts:
            return max(word_counts.items(), key=lambda item: item[1])[0]
        return None


    def _score_keywords(self, keywords, search_volume_data, keyword_difficulty_data, serp_data):
        """
        Score keywords for difficulty and opportunity using SerpAPI SV/KD and SERP signals.

        Args:
            keywords (list): List of keywords to score
            search_volume_data (dict): Dictionary of keyword: SV from SerpAPI
            keyword_difficulty_data (dict): Dictionary of keyword: KD from SerpAPI
            serp_data (dict): Full SERP data from SerpAPI

        Returns:
            dict: Scores for each keyword
        """
        logger.info("Scoring keywords using SV/KD and SERP signals.")
        scores = {}

        # Define weights for scoring factors (adjust based on desired emphasis)
        weight_sv = 0.4
        weight_kd = 0.4
        weight_serp_signals = 0.2 # Weight for insights from SERP features/competitors

        for keyword in keywords:
            # Get SV and KD from SerpAPI data (use 0 or default if not available)
            sv = search_volume_data.get(keyword)
            # SerpAPI KD is often a score out of 100, sometimes a string like "Hard"
            # Convert to a numerical scale 0-100 if it's not already
            kd_raw = keyword_difficulty_data.get(keyword)
            kd = None
            if isinstance(kd_raw, (int, float)):
                kd = max(0, min(100, int(kd_raw))) # Ensure it's within 0-100
            elif isinstance(kd_raw, str):
                 # Basic mapping for string difficulties if needed, refine based on SerpAPI output
                 kd_lower = kd_raw.lower()
                 if 'easy' in kd_lower: kd = 20
                 elif 'medium' in kd_lower: kd = 50
                 elif 'hard' in kd_lower: kd = 80
                 elif 'very hard' in kd_lower: kd = 95
                 else: kd = 50 # Default if unknown string
            else:
                kd = 50 # Default if no KD data

            # Estimate Search Volume score (simple scaling for now)
            # This needs refinement based on typical SV ranges in your niche
            # Max SV could be set based on expected range or highest SV found
            max_sv_in_data = max([v for v in search_volume_data.values() if isinstance(v, (int, float))] or [1], default=1) # Avoid division by zero
            sv_score = (sv / max_sv_in_data) * 100 if sv is not None and max_sv_in_data > 0 else 0
            sv_score = max(0, min(100, int(sv_score))) # Ensure 0-100 scale


            # Calculate score based on SERP signals (reusing some old logic, can be enhanced)
            serp_signal_score = 50 # Default neutral impact
            intent = self._classify_intent( # Re-classify or use already classified
                 keyword,
                 serp_data.get("features", {}).get(keyword, []),
                 serp_data.get("serp_data", {}).get(keyword, [])
                 )

            # Adjust based on SERP features (more features might mean more opportunity/complexity)
            features = serp_data.get("features", {}).get(keyword, [])
            serp_signal_score += min(len(features) * 3, 15) # Small boost for features


            # Adjust based on intent (Commercial/Transactional often higher opportunity/difficulty)
            if intent == "commercial":
                serp_signal_score += 5 # Slight opportunity boost
            elif intent == "transactional":
                serp_signal_score += 10 # More opportunity boost


            # Adjust based on competitor domains in top results (simplified, relies on SerpAPI organic_results)
            major_domains = ["wikipedia.org", "amazon.com", "youtube.com", "facebook.com", "linkedin.com", "twitter.com", "reddit.com"] # Expand this list
            top_domains = []
            for res in serp_data.get("serp_data", {}).get(keyword, [])[:5]:
                 if "link" in res:
                     domain = self._extract_domain(res["link"])
                     top_domains.append(domain)

            major_domain_count = sum(1 for domain in top_domains if any(major in domain for major in major_domains))
            serp_signal_score -= min(major_domain_count * 5, 20) # Deduct for major domain presence


            serp_signal_score = max(0, min(100, serp_signal_score)) # Ensure 0-100


            # Calculate overall scores using weighted average
            # Opportunity is primarily driven by SV and some SERP signals
            # Difficulty is primarily driven by KD and some SERP signals
            # Overall score is a balance, perhaps opportunity - difficulty
            opportunity = int((sv_score * 0.6) + (serp_signal_score * 0.4)) # Example weighting
            difficulty = int((kd * 0.7) + (serp_signal_score * 0.3)) # Example weighting

            opportunity = max(0, min(100, opportunity))
            difficulty = max(0, min(100, difficulty))

            # Combined score formula (opportunity favored, offset by difficulty)
            combined_score = int((opportunity * 0.7) - (difficulty * 0.3) + 50)
            combined_score = max(0, min(100, combined_score))


            scores[keyword] = {
                "search_volume": sv, # Include raw SV from API
                "keyword_difficulty": kd_raw, # Include raw KD from API
                "difficulty": difficulty, # Calculated Difficulty (0-100)
                "opportunity": opportunity, # Calculated Opportunity (0-100)
                "score": combined_score # Calculated Combined Score (0-100)
            }

        return scores

    def _extract_questions(self, paa_questions_data):
        """
        Extract question-based keywords from PAA data.

        Args:
            paa_questions_data (dict): Dictionary mapping keyword: [list of PAA questions]

        Returns:
            list: Unique question-based keywords
        """
        try:
            questions = []

            # Extract PAA questions from all keywords in the provided dict
            for keyword, paa_list in paa_questions_data.items():
                if isinstance(paa_list, list):
                     questions.extend(paa_list)

            # Note: Related searches are handled separately and can be added to recommendations
            # if they are in question format, but this method focuses on PAA specifically as requested.

            # Remove duplicates
            questions = list(dict.fromkeys(questions))

            return questions
        except Exception as e:
            logger.error(f"Error extracting questions: {str(e)}")
            return []


    def _extract_domain(self, url):
        """
        Extract domain from a URL - Keep existing method

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
             # Handle https://www.google.com/url?sa=E&source=gmail&q=googleusercontent.com for YouTube links
            if "https://www.google.com/url?sa=E&source=gmail&q=googleusercontent.com" in domain and "youtube.com" in url:
                 return "youtube.com"


            return domain
        except Exception as e:
            logger.error(f"Error extracting domain from URL '{url}': {str(e)}")
            # Return original URL if parsing fails
            return url