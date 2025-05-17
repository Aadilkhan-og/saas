import os
import json
import logging
from modules.gemini_client import GeminiClient

logger = logging.getLogger("keyword_research.insight_generator")

class InsightGenerator:
    """
    Generates insights from the collected keyword and competitor data
    """
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        try:
            gemini_client = GeminiClient(api_key=self.google_api_key, model="gemini-2.0-flash")
            self.llm = gemini_client.get_langchain_llm()
            logger.info("Gemini initialized for InsightGenerator")
        except Exception as e:
            logger.error(f"Error initializing Gemini: {str(e)}")
            self.llm = None
    
    def generate_insights(self, keyword_analysis, serp_data, competitor_data):
        """
        Generate insights from the collected data
        
        Args:
            keyword_analysis (dict): Processed keyword data
            serp_data (dict): SERP data for the keywords
            competitor_data (dict): Competitor content analysis
            
        Returns:
            dict: Generated insights
        """
        logger.info("Generating insights from the collected data")
        
        # Initialize insights
        insights = {
            "content_opportunities": [],
            "serp_feature_insights": [],
            "competitive_landscape": {},
            "keyword_recommendations": [],
            "topic_clusters": [],
            "intent_distribution": {},
            "summary": ""
        }
        
        try:
            # Generate content opportunities
            insights["content_opportunities"] = self._generate_content_opportunities(
                keyword_analysis, serp_data, competitor_data
            )
            logger.info(f"Generated {len(insights['content_opportunities'])} content opportunities")
            
            # Generate SERP feature insights
            insights["serp_feature_insights"] = self._generate_serp_feature_insights(
                serp_data
            )
            logger.info(f"Generated {len(insights['serp_feature_insights'])} SERP feature insights")
            
            # Analyze competitive landscape
            insights["competitive_landscape"] = self._analyze_competitive_landscape(
                serp_data, competitor_data
            )
            logger.info("Competitive landscape analysis completed")
            
            # Generate keyword recommendations
            insights["keyword_recommendations"] = self._generate_keyword_recommendations(
                keyword_analysis, serp_data
            )
            logger.info(f"Generated {len(insights['keyword_recommendations'])} keyword recommendations")
            
            # Extract topic clusters
            insights["topic_clusters"] = self._extract_topic_clusters(
                keyword_analysis
            )
            logger.info(f"Extracted {len(insights['topic_clusters'])} topic clusters")
            
            # Calculate intent distribution
            insights["intent_distribution"] = self._calculate_intent_distribution(
                keyword_analysis
            )
            logger.info("Intent distribution calculated")
            
            # Generate summary using LLM
            insights["summary"] = self._generate_summary_with_llm(
                keyword_analysis, serp_data, competitor_data, insights
            )
            logger.info("Summary generated")
            
            return insights
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}", exc_info=True)
            # Return partial insights
            return insights
    
    def _generate_content_opportunities(self, keyword_analysis, serp_data, competitor_data):
        """
        Generate content opportunities based on the data
        
        Args:
            keyword_analysis (dict): Processed keyword data
            serp_data (dict): SERP data for the keywords
            competitor_data (dict): Competitor content analysis
            
        Returns:
            list: Content opportunities
        """
        try:
            opportunities = []
            
            # 1. Check for question-based content opportunities
            if keyword_analysis.get("question_keywords"):
                questions = keyword_analysis["question_keywords"][:5]  # Top 5 questions
                
                if questions:
                    opportunities.append({
                        "type": "question_content",
                        "title": "Create FAQ Content",
                        "description": f"Create FAQ content addressing these top questions: {', '.join(questions)}",
                        "keywords": questions
                    })
                    logger.debug("Added question-based content opportunity")
            
            # 2. Check for content type gaps based on competitor analysis
            if competitor_data.get("summary", {}).get("most_common_content_type"):
                most_common_type = competitor_data["summary"]["most_common_content_type"]
                
                # Suggest alternative content types
                if most_common_type == "blog article":
                    opportunities.append({
                        "type": "content_type_gap",
                        "title": "Create Interactive Content",
                        "description": "Competitors mainly use blog articles. Consider creating interactive tools or calculators to differentiate.",
                        "content_type": "interactive"
                    })
                    logger.debug("Added interactive content opportunity (gap from blog articles)")
                elif most_common_type == "product page":
                    opportunities.append({
                        "type": "content_type_gap",
                        "title": "Create Comprehensive Guide",
                        "description": "Competitors focus on product pages. Consider creating in-depth guides to capture informational intent.",
                        "content_type": "guide"
                    })
                    logger.debug("Added guide content opportunity (gap from product pages)")
                elif most_common_type == "landing page":
                    opportunities.append({
                        "type": "content_type_gap",
                        "title": "Create Comparison Content",
                        "description": "Competitors focus on landing pages. Create detailed comparison content to help users in their decision process.",
                        "content_type": "comparison"
                    })
                    logger.debug("Added comparison content opportunity (gap from landing pages)")
            
            # 3. Check for intent gaps
            intent_distribution = self._calculate_intent_distribution(keyword_analysis)
            
            # If informational intent is high but commercial content dominates
            if intent_distribution.get("informational", 0) > 0.4 and competitor_data.get("summary", {}).get("most_common_content_type") in ["product page", "landing page"]:
                opportunities.append({
                    "type": "intent_gap",
                    "title": "Create Educational Content",
                    "description": "High informational intent but commercial content dominates. Create educational content to capture this intent.",
                    "intent": "informational"
                })
                logger.debug("Added educational content opportunity (intent gap)")
            
            # If commercial intent is high but mostly informational content available
            elif intent_distribution.get("commercial", 0) > 0.3 and competitor_data.get("summary", {}).get("most_common_content_type") in ["blog article", "guide"]:
                opportunities.append({
                    "type": "intent_gap",
                    "title": "Create Product Comparison Content",
                    "description": "High commercial intent but mostly informational content available. Create product comparisons and reviews.",
                    "intent": "commercial"
                })
                logger.debug("Added product comparison opportunity (intent gap)")
            
            # 4. Opportunity for featured snippet targeting
            featured_snippet_keywords = []
            for keyword, features in serp_data.get("features", {}).items():
                if "featured_snippet" in features:
                    featured_snippet_keywords.append(keyword)
            
            if featured_snippet_keywords:
                opportunities.append({
                    "type": "serp_feature_opportunity",
                    "title": "Target Featured Snippets",
                    "description": f"These keywords have featured snippets that can be targeted: {', '.join(featured_snippet_keywords)}",
                    "keywords": featured_snippet_keywords
                })
                logger.debug(f"Added featured snippet opportunity for {len(featured_snippet_keywords)} keywords")
            
            # 5. Content freshness opportunity
            if competitor_data.get("summary", {}).get("content_freshness", "old") == "old":
                opportunities.append({
                    "type": "freshness_opportunity",
                    "title": "Create Updated Content",
                    "description": "Competitor content is outdated. Create fresh, up-to-date content with current information.",
                    "strategy": "freshness"
                })
                logger.debug("Added content freshness opportunity")
            
            # 6. Content depth opportunity
            avg_word_count = competitor_data.get("summary", {}).get("avg_word_count", 0)
            if avg_word_count > 0:
                if avg_word_count < 800:
                    opportunities.append({
                        "type": "depth_opportunity",
                        "title": "Create In-depth Content",
                        "description": f"Competitor content averages only {avg_word_count} words. Create more comprehensive content to stand out.",
                        "strategy": "depth"
                    })
                    logger.debug("Added content depth opportunity (short competitor content)")
                elif avg_word_count > 2500:
                    opportunities.append({
                        "type": "depth_opportunity",
                        "title": "Create Concise Content",
                        "description": f"Competitor content averages {avg_word_count} words. Create more concise, focused content as an alternative.",
                        "strategy": "conciseness"
                    })
                    logger.debug("Added content conciseness opportunity (long competitor content)")
            
            # 7. Video content opportunity
            video_keywords = []
            for keyword, features in serp_data.get("features", {}).items():
                if "video_carousel" in features:
                    video_keywords.append(keyword)
            
            if video_keywords:
                opportunities.append({
                    "type": "media_opportunity",
                    "title": "Create Video Content",
                    "description": f"These keywords show video results: {', '.join(video_keywords)}. Create video content to capture this traffic.",
                    "keywords": video_keywords,
                    "media_type": "video"
                })
                logger.debug(f"Added video content opportunity for {len(video_keywords)} keywords")
            
            return opportunities
        except Exception as e:
            logger.error(f"Error generating content opportunities: {str(e)}")
            return []
    
    def _generate_serp_feature_insights(self, serp_data):
        """
        Generate insights about SERP features
        
        Args:
            serp_data (dict): SERP data for the keywords
            
        Returns:
            list: SERP feature insights
        """
        try:
            insights = []
            
            # Count feature occurrences across all keywords
            feature_counts = {}
            for keyword, features in serp_data.get("features", {}).items():
                for feature in features:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            # Generate insights for common features
            for feature, count in feature_counts.items():
                if count >= 2:  # Feature appears multiple times
                    percentage = round((count / len(serp_data.get("features", {}))) * 100 if serp_data.get("features") else 0)
                    description = ""
                    strategy = ""
                    
                    if feature == "video_carousel":
                        description = f"Video results appear for {percentage}% of keywords - indicating video content opportunity."
                        strategy = "Create video content to capture this SERP real estate. Focus on tutorials, how-to's, and demonstrations."
                    elif feature == "featured_snippet":
                        description = f"Featured snippets appear for {percentage}% of keywords - optimize for position zero."
                        strategy = "Structure content with clear definitions, steps, and concise answers to common questions."
                    elif feature == "image_pack":
                        description = f"Image packs appear for {percentage}% of keywords - visual content opportunity."
                        strategy = "Include high-quality, relevant images in your content with proper alt text and schema markup."
                    elif feature == "shopping_results":
                        description = f"Shopping results appear for {percentage}% of keywords - strong commercial intent."
                        strategy = "Optimize product content with structured data and competitive pricing information."
                    elif feature == "knowledge_panel":
                        description = f"Knowledge panels appear for {percentage}% of keywords - entity optimization opportunity."
                        strategy = "Implement entity-based SEO and schema markup to enhance brand visibility."
                    elif feature == "local_pack":
                        description = f"Local results appear for {percentage}% of keywords - local SEO opportunity."
                        strategy = "Optimize Google Business Profile and implement local-focused content strategy."
                    elif feature == "people_also_ask":
                        description = f"People Also Ask boxes appear for {percentage}% of keywords - question-based content opportunity."
                        strategy = "Create FAQ content that directly answers these related questions."
                    else:
                        description = f"{feature} appears in {percentage}% of search results."
                        strategy = "Monitor this feature and determine how to optimize for it."
                    
                    insights.append({
                        "feature": feature,
                        "occurrence": count,
                        "percentage": percentage,
                        "description": description,
                        "strategy": strategy
                    })
                    logger.debug(f"Added SERP feature insight for {feature}")
            
            # Sort insights by occurrence (highest first)
            insights.sort(key=lambda x: x["occurrence"], reverse=True)
            
            return insights
        except Exception as e:
            logger.error(f"Error generating SERP feature insights: {str(e)}")
            return []
    
    def _analyze_competitive_landscape(self, serp_data, competitor_data):
        """
        Analyze the competitive landscape
        
        Args:
            serp_data (dict): SERP data for the keywords
            competitor_data (dict): Competitor content analysis
            
        Returns:
            dict: Competitive landscape analysis
        """
        try:
            # Initialize analysis
            analysis = {
                "domain_distribution": {},
                "content_length_avg": 0,
                "dominant_content_types": [],
                "common_themes": [],
                "domain_authority_level": "medium",
                "content_freshness": "mixed",
                "content_quality": "medium"
            }
            
            # Count domain occurrences in SERP data
            domain_counts = {}
            total_domains = 0
            
            for keyword, results in serp_data.get("serp_data", {}).items():
                for result in results:
                    if "url" in result:
                        # Extract domain from URL
                        domain = self._extract_domain(result["url"])
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                        total_domains += 1
            
            # Calculate domain distribution
            for domain, count in domain_counts.items():
                if count > 1:  # Only include domains that appear multiple times
                    percentage = count / total_domains if total_domains > 0 else 0
                    analysis["domain_distribution"][domain] = round(percentage * 100)
            
            # Sort domain distribution by frequency
            analysis["domain_distribution"] = dict(sorted(
                analysis["domain_distribution"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])  # Top 10 domains
            
            # Get content length average from competitor data
            if competitor_data.get("summary", {}).get("avg_word_count"):
                analysis["content_length_avg"] = competitor_data["summary"]["avg_word_count"]
            
            # Get dominant content types
            if competitor_data.get("summary", {}).get("content_types"):
                content_types = competitor_data["summary"]["content_types"]
                analysis["dominant_content_types"] = [
                    {"type": content_type, "count": count}
                    for content_type, count in content_types.items()
                ]
            
            # Get common themes
            if competitor_data.get("common_themes"):
                analysis["common_themes"] = competitor_data["common_themes"]
            
            # Determine domain authority level
            major_domains = ["wikipedia.org", "amazon.com", "youtube.com", "facebook.com", "linkedin.com"]
            major_domain_percentage = sum(
                analysis["domain_distribution"].get(domain, 0) 
                for domain in analysis["domain_distribution"] 
                if any(major in domain for major in major_domains)
            )
            
            if major_domain_percentage > 40:
                analysis["domain_authority_level"] = "high"
            elif major_domain_percentage > 20:
                analysis["domain_authority_level"] = "medium"
            else:
                analysis["domain_authority_level"] = "low"
            
            # Assess content freshness and quality
            # These would normally be extracted from the competitor content, but for now use placeholders
            analysis["content_freshness"] = "mixed"
            analysis["content_quality"] = "medium"
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing competitive landscape: {str(e)}")
            return {
                "domain_distribution": {},
                "content_length_avg": 0,
                "dominant_content_types": [],
                "common_themes": [],
                "domain_authority_level": "unknown",
                "content_freshness": "unknown",
                "content_quality": "unknown"
            }
    
    def _generate_keyword_recommendations(self, keyword_analysis, serp_data):
        """
        Generate keyword recommendations based on the analysis
        
        Args:
            keyword_analysis (dict): Processed keyword data
            serp_data (dict): SERP data for the keywords
            
        Returns:
            list: Keyword recommendations
        """
        try:
            recommendations = []
            
            # 1. Recommend keywords with high opportunity and lower difficulty
            if keyword_analysis.get("keyword_scores"):
                scores = keyword_analysis["keyword_scores"]
                
                # Find keywords with high opportunity score
                high_opportunity_keywords = []
                for keyword, score_data in scores.items():
                    if score_data.get("opportunity", 0) > 60 and score_data.get("difficulty", 100) < 70:
                        high_opportunity_keywords.append({
                            "keyword": keyword,
                            "score": score_data.get("score", 0),
                            "intent": keyword_analysis.get("intent_classification", {}).get(keyword, "unknown")
                        })
                
                # Sort by score (highest first)
                high_opportunity_keywords.sort(key=lambda x: x["score"], reverse=True)
                
                # Add to recommendations
                for kw_data in high_opportunity_keywords[:5]:  # Top 5 opportunities
                    recommendations.append({
                        "type": "high_opportunity",
                        "keyword": kw_data["keyword"],
                        "reason": f"High opportunity with manageable difficulty (Intent: {kw_data['intent']})"
                    })
                    logger.debug(f"Added high opportunity keyword recommendation: {kw_data['keyword']}")
            
            # 2. Recommend question keywords
            if keyword_analysis.get("question_keywords"):
                for question in keyword_analysis["question_keywords"][:3]:  # Top 3 questions
                    recommendations.append({
                        "type": "question",
                        "keyword": question,
                        "reason": "Question-based keywords often have featured snippet opportunities"
                    })
                    logger.debug(f"Added question keyword recommendation: {question}")
            
            # 3. Recommend keywords from related searches
            related_searches = []
            for keyword, searches in serp_data.get("related_searches", {}).items():
                related_searches.extend(searches)
            
            # Remove duplicates and limit to top 3
            unique_related = list(dict.fromkeys(related_searches))[:3]
            
            for related in unique_related:
                recommendations.append({
                    "type": "related_search",
                    "keyword": related,
                    "reason": "Found in related searches, indicating user interest"
                })
                logger.debug(f"Added related search keyword recommendation: {related}")
            
            # 4. Recommend keywords by intent type (if missing from current set)
            intent_distribution = self._calculate_intent_distribution(keyword_analysis)
            
            # If informational intent is low, recommend some informational keywords
            if intent_distribution.get("informational", 0) < 0.2 and serp_data.get("paa_questions"):
                for keyword, questions in serp_data.get("paa_questions", {}).items():
                    if questions:
                        recommendations.append({
                            "type": "intent_balance",
                            "keyword": questions[0],
                            "reason": "Adds informational content to balance your keyword portfolio"
                        })
                        logger.debug(f"Added informational keyword recommendation: {questions[0]}")
                        break
            
            # If commercial intent is low but relevant, recommend some commercial keywords
            if intent_distribution.get("commercial", 0) < 0.2:
                for keyword in keyword_analysis.get("keywords", []):
                    if "review" in keyword or "best" in keyword or "vs" in keyword:
                        recommendations.append({
                            "type": "intent_balance",
                            "keyword": keyword,
                            "reason": "Adds commercial content to balance your keyword portfolio"
                        })
                        logger.debug(f"Added commercial keyword recommendation: {keyword}")
                        break
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating keyword recommendations: {str(e)}")
            return []
    
    def _extract_topic_clusters(self, keyword_analysis):
        """
        Extract topic clusters from the keyword analysis
        
        Args:
            keyword_analysis (dict): Processed keyword data
            
        Returns:
            list: Topic clusters
        """
        try:
            # Return clusters directly from keyword analysis if available
            return keyword_analysis.get("clusters", [])
        except Exception as e:
            logger.error(f"Error extracting topic clusters: {str(e)}")
            return []
    
    def _calculate_intent_distribution(self, keyword_analysis):
        """
        Calculate the distribution of intent across keywords
        
        Args:
            keyword_analysis (dict): Processed keyword data
            
        Returns:
            dict: Intent distribution
        """
        try:
            intent_counts = {}
            total_keywords = 0
            
            # Count intents
            for keyword, intent in keyword_analysis.get("intent_classification", {}).items():
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                total_keywords += 1
            
            # Calculate distribution
            distribution = {}
            for intent, count in intent_counts.items():
                distribution[intent] = round(count / total_keywords, 2) if total_keywords > 0 else 0
            
            return distribution
        except Exception as e:
            logger.error(f"Error calculating intent distribution: {str(e)}")
            return {"informational": 0.25, "commercial": 0.25, "transactional": 0.25, "navigational": 0.25}
    
    def _generate_summary_with_llm(self, keyword_analysis, serp_data, competitor_data, insights):
        """
        Generate a summary of insights using the LLM
        
        Args:
            keyword_analysis (dict): Processed keyword data
            serp_data (dict): SERP data for the keywords
            competitor_data (dict): Competitor content analysis
            insights (dict): Generated insights
            
        Returns:
            str: Summary of insights
        """
        try:
            # If LLM isn't available, generate a static summary
            if not self.llm:
                return self._generate_static_summary(keyword_analysis, serp_data, competitor_data, insights)
            
            # For future improvement: Use the LLM to generate a dynamic summary
            # This would require formatting the data for the LLM and calling it
            
            # For now, return the static summary
            return self._generate_static_summary(keyword_analysis, serp_data, competitor_data, insights)
        except Exception as e:
            logger.error(f"Error generating summary with LLM: {str(e)}")
            return "Analysis complete. See detailed results below."
    
    def _generate_static_summary(self, keyword_analysis, serp_data, competitor_data, insights):
        """
        Generate a static summary of insights
        
        Args:
            keyword_analysis (dict): Processed keyword data
            serp_data (dict): SERP data for the keywords
            competitor_data (dict): Competitor content analysis
            insights (dict): Generated insights
            
        Returns:
            str: Summary of insights
        """
        try:
            # Count keywords by intent
            intent_counts = {}
            for intent, percentage in insights.get("intent_distribution", {}).items():
                intent_counts[intent] = int(percentage * 100) if percentage else 0
            
            # Get most common intent
            most_common_intent = max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else "informational"
            
            # Check content types
            content_types = [ct["type"] for ct in insights.get("competitive_landscape", {}).get("dominant_content_types", [])]
            content_type_str = ", ".join(content_types[:2]) if content_types else "various"
            
            # Get domain authority level
            domain_authority = insights.get("competitive_landscape", {}).get("domain_authority_level", "medium")
            
            # Calculate opportunity score
            opportunity_scores = [data.get("opportunity", 0) for data in keyword_analysis.get("keyword_scores", {}).values()]
            avg_opportunity = sum(opportunity_scores) / len(opportunity_scores) if opportunity_scores else 50
            
            # Create summary
            summary = f"""
            Based on the analysis of {len(keyword_analysis.get('intent_classification', {}))} keywords, the most common intent is {most_common_intent} ({intent_counts.get(most_common_intent, 0)}%).
            
            The competitive landscape shows {domain_authority} domain authority with content primarily in {content_type_str} format at an average length of {insights.get('competitive_landscape', {}).get('content_length_avg', 0)} words.
            
            Key opportunities include {len(insights.get('content_opportunities', []))} content gaps to address and {len(insights.get('keyword_recommendations', []))} recommended keywords to target, with an average opportunity score of {int(avg_opportunity)}/100.
            
            SERP analysis reveals {len(insights.get('serp_feature_insights', []))} notable SERP features that can be optimized for better visibility.
            """
            
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating static summary: {str(e)}")
            return "Analysis complete. See detailed results below."
    
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
