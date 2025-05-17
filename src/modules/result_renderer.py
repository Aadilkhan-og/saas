# Filename: src/modules/result_renderer.py
import logging
import json
from datetime import datetime

logger = logging.getLogger("keyword_research.result_renderer")

class ResultRenderer:
    """
    Renders the final results in a structured format
    """

    def __init__(self):
        pass

    # Updated signature to accept full serp_data and competitor_data for detailed view
    def render_results(self, input_data, keyword_analysis, serp_data, competitor_data, insights, raw_serp_data, raw_competitor_data):
        """
        Render the final results from all collected and analyzed data

        Args:
            input_data (dict): Processed user input
            keyword_analysis (dict): Processed keyword data (intent, scores, clusters, questions)
            serp_data (dict): Aggregated SERP data (used for insights like domain distribution, features)
            competitor_data (dict): Aggregated Competitor content analysis (used for insights like avg word count, dominant types)
            insights (dict): Generated insights (opportunities, recommendations, summaries)
            raw_serp_data (dict): Full, raw SERP data collected by SerpCollector
            raw_competitor_data (dict): Full, raw Competitor content analysis data by ContentAnalyzer

        Returns:
            dict: Structured results for presentation, including detailed data
        """
        logger.info("Rendering final results, including detailed data")

        try:
            # Prepare the result structure
            results = {
                "summary": {
                    "title": f"Keyword Research: {input_data.get('seed_keyword', 'Unknown')}",
                    "date": self._get_current_date(),
                    "keyword_count": len(input_data.get('keywords', [])),
                    "insight_summary": insights.get('summary', ''),
                    "top_opportunities": self._extract_top_opportunities(insights)
                },
                "keywords": {
                    "analyzed": self._format_keywords(input_data.get('keywords', []), keyword_analysis),
                    "questions": keyword_analysis.get('question_keywords', []),
                    "opportunities": self._extract_keyword_opportunities(keyword_analysis, insights)
                },
                "intent_analysis": {
                    "distribution": insights.get('intent_distribution', {}),
                    "clusters": insights.get('topic_clusters', [])
                },
                "serp_insights": {
                    "features": insights.get('serp_feature_insights', []),
                    "top_competitors": self._extract_top_competitors(insights)
                },
                # Add the detailed SERP data here using raw_serp_data
                "detailed_serp_results": {
                    "keyword": raw_serp_data.get('keywords', [None])[0], # Assuming the first keyword is the main one
                    "results": raw_serp_data.get('serp_data', {}).get(raw_serp_data.get('keywords', [None])[0], []), # Get detailed results for the first keyword
                    "featured_snippet": raw_serp_data.get('featured_snippet'), # Note: SerpCollector's raw_serp_data structure might need adjustment if it's per keyword
                    "paa_questions": raw_serp_data.get('paa_questions', {}).get(raw_serp_data.get('keywords', [None])[0], []), # Get PAA for the first keyword
                    "related_searches": raw_serp_data.get('related_searches', {}).get(raw_serp_data.get('keywords', [None])[0], []), # Get related searches for the first keyword
                    "features": raw_serp_data.get('features', {}).get(raw_serp_data.get('keywords', [None])[0], []), # Get features for the first keyword
                    # If SerpCollector's raw_serp_data structure is different (e.g., has a single combined list of results/paa/etc.) adjust parsing here
                    # Example: If raw_serp_data was { 'results': [], 'paa_questions': [], ... } directly, you'd use raw_serp_data.get('results', []) etc.
                },


                "content_strategy": {
                    "opportunities": insights.get('content_opportunities', []),
                    "recommendations": self._generate_content_recommendations(keyword_analysis, insights, raw_competitor_data), # Use raw_competitor_data for recommendations if needed
                    "competitor_content": { # Keep aggregated summary for overview
                        "word_count_avg": insights.get('competitive_landscape', {}).get('content_length_avg', 0),
                        "content_types": insights.get('competitive_landscape', {}).get('dominant_content_types', []),
                        "themes": insights.get('competitive_landscape', {}).get('common_themes', [])
                    },
                },
                 # Add the detailed competitor content analysis here using raw_competitor_data
                "detailed_content_analysis": self._format_detailed_content_analysis(raw_competitor_data),


                "next_steps": self._generate_next_steps(keyword_analysis, insights),
            }

            return results
        except Exception as e:
            logger.error(f"Error rendering results: {str(e)}", exc_info=True)
            # Return minimal results on error
            return {
                "summary": {
                    "title": f"Keyword Research: {input_data.get('seed_keyword', 'Unknown')}",
                    "date": self._get_current_date(),
                    "keyword_count": len(input_data.get('keywords', [])),
                    "insight_summary": "An error occurred while rendering results.",
                    "top_opportunities": []
                },
                "error": str(e)
            }

    def _format_keywords(self, keywords, keyword_analysis):
        """
        Format keywords with their analysis data

        Args:
            keywords (list): List of analyzed keywords
            keyword_analysis (dict): Processed keyword data

        Returns:
            list: Formatted keyword data
        """
        try:
            formatted_keywords = []

            # Ensure keyword_analysis has the necessary keys
            intent_classification = keyword_analysis.get('intent_classification', {})
            keyword_scores = keyword_analysis.get('keyword_scores', {})

            for keyword in keywords:
                keyword_data = {
                    "keyword": keyword,
                    "intent": intent_classification.get(keyword, "unknown"),
                    "scores": keyword_scores.get(keyword, {"difficulty": 50, "opportunity": 50, "score": 50}) # Provide default scores
                }

                formatted_keywords.append(keyword_data)

            # Sort by score if available
            formatted_keywords.sort(
                key=lambda x: x.get('scores', {}).get('score', 0),
                reverse=True
            )

            return formatted_keywords
        except Exception as e:
            logger.error(f"Error formatting keywords: {str(e)}")
            return [{"keyword": kw, "intent": "unknown", "scores": {"difficulty": 50, "opportunity": 50, "score": 50}} for kw in keywords]

    def _extract_top_opportunities(self, insights):
        """
        Extract top opportunities from insights

        Args:
            insights (dict): Generated insights

        Returns:
            list: Top opportunities
        """
        try:
            # Combine content opportunities and keyword recommendations and SERP feature insights
            all_opportunities = []

            # Add content opportunities
            for opportunity in insights.get('content_opportunities', []): # Include all content opportunities
                all_opportunities.append({
                    "type": "content",
                    "title": opportunity.get('title', ''),
                    "description": opportunity.get('description', '')
                })

            # Add keyword recommendations
            for recommendation in insights.get('keyword_recommendations', []): # Include all keyword recommendations
                 # Filter out recommendations that are just the seed keyword itself if it has no special reason
                 if recommendation.get('type') == 'high_opportunity' and recommendation.get('keyword') == insights.get('summary', {}).get('title', '').replace('Keyword Research: ', '').strip() and not recommendation.get('reason'):
                     continue
                 all_opportunities.append({
                    "type": "keyword",
                    "title": f"Target: {recommendation.get('keyword', '')}",
                    "description": recommendation.get('reason', '')
                })

            # Add SERP feature insights
            for insight in insights.get('serp_feature_insights', []): # Include all SERP feature insights
                all_opportunities.append({
                    "type": "serp_feature",
                    "title": f"Optimize for {insight.get('feature', '').replace('_', ' ')}",
                    "description": insight.get('description', '')
                })

            # Return top 5 opportunities (can adjust number)
            return all_opportunities[:5]
        except Exception as e:
            logger.error(f"Error extracting top opportunities: {str(e)}")
            return []

    def _extract_keyword_opportunities(self, keyword_analysis, insights):
        """
        Extract keyword opportunities from the analysis

        Args:
            keyword_analysis (dict): Processed keyword data
            insights (dict): Generated insights

        Returns:
            list: Keyword opportunities
        """
        try:
            opportunities = []

            # Add keywords with high opportunity score from keyword_scores
            if keyword_analysis.get('keyword_scores'):
                scores = keyword_analysis['keyword_scores']

                # Find keywords with high opportunity score
                for keyword, score_data in scores.items():
                    # Only add if opportunity is high or score is high
                    if score_data.get('opportunity', 0) > 60 or score_data.get('score', 0) > 70:
                        opportunities.append({
                            "keyword": keyword,
                            "score": score_data.get('score', 0),
                            "difficulty": score_data.get('difficulty', 50),
                            "opportunity": score_data.get('opportunity', 50),
                            "intent": keyword_analysis.get('intent_classification', {}).get(keyword, "unknown"),
                            "type": "high_opportunity"
                        })

            # Add keywords from recommendations if they aren't already in high_opportunity (based on keyword string)
            recommended_keywords = {opp['keyword'] for opp in opportunities if opp.get('type') == 'high_opportunity'}
            for recommendation in insights.get('keyword_recommendations', []):
                 kw = recommendation.get('keyword')
                 if kw and kw not in recommended_keywords:
                      # Find the score for this recommended keyword if available
                      score_data = keyword_analysis.get('keyword_scores', {}).get(kw, {"difficulty": 50, "opportunity": 50, "score": 50})
                      opportunities.append({
                           "keyword": kw,
                           "score": score_data.get('score', 0),
                           "difficulty": score_data.get('difficulty', 50),
                           "opportunity": score_data.get('opportunity', 50),
                           "intent": keyword_analysis.get('intent_classification', {}).get(kw, "unknown"),
                           "type": recommendation.get('type', 'recommended'), # Use recommendation type
                           "reason": recommendation.get('reason', '') # Include reason
                      })


            # Sort by score and take top 10 (can adjust number)
            opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)

            return opportunities[:10]
        except Exception as e:
            logger.error(f"Error extracting keyword opportunities: {str(e)}")
            return []

    def _extract_top_competitors(self, insights):
        """
        Extract top competitors from insights

        Args:
            insights (dict): Generated insights

        Returns:
            list: Top competitors
        """
        try:
            competitors = []

            # Extract domain distribution
            domain_distribution = insights.get('competitive_landscape', {}).get('domain_distribution', {})

            for domain, percentage in domain_distribution.items():
                competitors.append({
                    "domain": domain,
                    "percentage": percentage
                })

            # Sort by percentage
            competitors.sort(key=lambda x: x.get('percentage', 0), reverse=True)

            # Return top 5 (can adjust number)
            return competitors[:5]
        except Exception as e:
            logger.error(f"Error extracting top competitors: {str(e)}")
            return []

    # --- Detailed Content Analysis Formatting ---
    def _format_detailed_content_analysis(self, raw_competitor_data):
        """
        Formats the detailed competitor content analysis data for the report.

        Args:
            raw_competitor_data (dict): Full, raw Competitor content analysis data.

        Returns:
            list: List of formatted analysis results per URL.
        """
        formatted_list = []
        # raw_competitor_data['content_analysis'] holds the dict {url: {analysis_data}}
        content_analysis_by_url = raw_competitor_data.get('content_analysis', {})
        analyzed_urls = raw_competitor_data.get('analyzed_urls', []) # Use the list of successfully analyzed URLs

        # Iterate through the URLs that were actually analyzed
        for url in analyzed_urls:
            analysis_data = content_analysis_by_url.get(url)
            if analysis_data: # Should exist if in analyzed_urls, but double-check
                formatted_list.append({
                    "url": url,
                    "analysis": analysis_data # Include the full analysis data for this URL
                })

        return formatted_list


    def _generate_content_recommendations(self, keyword_analysis, insights, competitor_data):
        """
        Generate content recommendations based on the analysis

        Args:
            keyword_analysis (dict): Processed keyword data
            insights (dict): Generated insights
            competitor_data (dict): Full competitor content analysis data (used here for deeper checks if needed)

        Returns:
            list: Content recommendations
        """
        try:
            recommendations = []

            # Add content length recommendation
            avg_word_count = insights.get('competitive_landscape', {}).get('content_length_avg', 0)

            if avg_word_count > 0:
                # Recommend aiming slightly higher than average
                target_word_count = ((avg_word_count + 200) // 100) * 100
                if target_word_count <= avg_word_count: # Ensure it's at least 100 words higher
                     target_word_count = avg_word_count + 100

                recommendations.append({
                    "type": "content_length",
                    "title": f"Create content with {target_word_count}+ words",
                    "description": f"Competitor content averages {avg_word_count} words. Aim for greater depth to stand out."
                })

            # Add content type recommendation
            # Use the dominant type from the aggregated insights
            dominant_content_type_list = insights.get('competitive_landscape', {}).get('dominant_content_types', [])
            dominant_content_type = dominant_content_type_list[0]['type'] if dominant_content_type_list else 'unknown'


            if dominant_content_type and dominant_content_type != 'unknown':
                # Suggest alternative or enhanced content based on the dominant type
                if dominant_content_type.lower() in ["blog article", "informative web page", "guide", "article"]:
                    recommendations.append({
                        "type": "content_format",
                        "title": "Create interactive content or detailed tools",
                        "description": f"Competitors mainly use {dominant_content_type}. Consider creating interactive tools, calculators, or in-depth resources to differentiate."
                    })
                elif dominant_content_type.lower() in ["product page", "landing page", "homepage"]:
                    recommendations.append({
                        "type": "content_format",
                        "title": "Create comparison or review content",
                        "description": f"Competitors focus on {dominant_content_type}. Develop detailed comparison content or product reviews to capture commercial intent."
                    })
                elif dominant_content_type.lower() in ["video page", "video"]:
                     recommendations.append({
                        "type": "content_format",
                        "title": "Create written guides or infographics",
                        "description": f"Competitors focus on video content. Offer alternative formats like detailed written guides, blog posts, or visual infographics."
                    })
                else: # Generic diversification recommendation
                     recommendations.append({
                        "type": "content_format",
                        "title": "Diversify content formats",
                        "description": f"Explore content types beyond the dominant {dominant_content_type} format, such as videos, infographics, or tools."
                    })


            # Add recommendation based on intent distribution
            intent_distribution = insights.get('intent_distribution', {})

            # Find the highest intent percentage
            if intent_distribution:
                sorted_intents = sorted(intent_distribution.items(), key=lambda x: x[1], reverse=True)
                # Check if there's a clear dominant intent (e.g., > 50%)
                if sorted_intents and sorted_intents[0][1] > 0.5:
                    highest_intent = sorted_intents[0][0]

                    # Check if the dominant content type aligns with the highest intent.
                    # If not, recommend creating content for that intent.
                    intent_content_mapping = {
                        "informational": ["blog article", "guide", "tutorial", "support article", "informative web page"],
                        "commercial": ["review", "comparison", "product page", "landing page"],
                        "transactional": ["product page", "landing page", "checkout page"], # Less likely to analyze checkout pages
                        "navigational": ["homepage", "about page", "contact page"],
                    }

                    dominant_type_matches_intent = False
                    if highest_intent in intent_content_mapping:
                        # Check if the dominant content type (string) is related to the expected types for this intent
                        # This is a simplification; a more complex check against all analyzed content types would be better
                        if dominant_content_type.lower() in [t.lower() for t in intent_content_mapping[highest_intent]]:
                            dominant_type_matches_intent = True

                    # If dominant content type doesn't align strongly with highest intent
                    if not dominant_type_matches_intent:
                         recommendations.append({
                             "type": "intent_targeting",
                             "title": f"Create content for {highest_intent} intent",
                             "description": f"The primary intent for these keywords is {highest_intent}, but competitor content doesn't strongly reflect this. Create content tailored to this intent."
                         })
                    # Else, if it does align, reinforce focusing on that intent
                    elif dominant_type_matches_intent and highest_intent != 'unknown':
                         recommendations.append({
                             "type": "intent_targeting",
                             "title": f"Reinforce content for {highest_intent} intent",
                             "description": f"Competitor content aligns with the dominant {highest_intent} intent. Ensure your content strategy strongly addresses this."
                         })


            # Add SERP feature-based recommendation (from insights)
            serp_insights_list = insights.get('serp_feature_insights', [])

            # Recommend optimizing for common features
            for insight in serp_insights_list[:3]: # Top 3 features
                 recommendations.append({
                     "type": "serp_feature",
                     "title": f"Optimize for {insight.get('feature', '').replace('_', ' ')}",
                     "description": insight.get('strategy', f"This feature appears in {insight.get('percentage', 0)}% of results for these keywords.")
                 })


            # Add Content Freshness recommendation if data is available and indicates old content
            # This assumes competitor_data has a 'content_freshness' summary key
            content_freshness = competitor_data.get('summary', {}).get('content_freshness')
            if content_freshness and content_freshness != 'unknown':
                 if content_freshness == 'old':
                     recommendations.append({
                         "type": "freshness_opportunity",
                         "title": "Update or create fresh content",
                         "description": "Competitor content for these keywords appears outdated. Opportunity to create fresh, up-to-date resources."
                     })
                 elif content_freshness == 'mixed':
                      recommendations.append({
                         "type": "freshness_opportunity",
                         "title": "Analyze content freshness",
                         "description": "Content freshness is mixed among competitors. Identify older top-ranking content you can improve and update."
                     })


            # Ensure unique recommendations based on title/description combination
            unique_recommendations = {}
            for rec in recommendations:
                key = (rec.get('title'), rec.get('description'))
                if key not in unique_recommendations:
                    unique_recommendations[key] = rec

            return list(unique_recommendations.values())


        except Exception as e:
            logger.error(f"Error generating content recommendations: {str(e)}")
            return []

    def _generate_next_steps(self, keyword_analysis, insights):
        """
        Generate recommended next steps based on the analysis

        Args:
            keyword_analysis (dict): Processed keyword data
            insights (dict): Generated insights

        Returns:
            list: Recommended next steps
        """
        try:
            next_steps = []

            # Add standard next steps
            next_steps.append({
                "step": 1,
                "title": "Review keyword opportunities",
                "description": "Select primary and secondary target keywords from the 'Top Keywords by Opportunity Score' and 'Question-Based Keywords' lists based on your business goals and resources."
            })

            next_steps.append({
                "step": 2,
                "title": "Analyze competitor content details",
                "description": "Review the 'Detailed Competitor Content Analysis' section to understand the structure, themes, and media types used by top-ranking pages for your chosen keywords. Identify what works and potential content gaps."
            })


            # Add specific steps based on identified opportunities
            content_opportunities = insights.get('content_opportunities', [])
            serp_insights_list = insights.get('serp_feature_insights', [])
            keyword_recommendations_list = insights.get('keyword_recommendations', [])

            # Step for creating FAQ content if question keywords were found
            if keyword_analysis.get("question_keywords"):
                 next_steps.append({
                    "step": "2a", # Use a letter step to insert
                    "title": "Create FAQ content",
                    "description": "Develop dedicated FAQ sections or blog posts that directly answer the 'People Also Ask' and question-based keywords identified."
                })

            # Step for content type gap if identified
            if any(opp.get('type') == 'content_type_gap' for opp in content_opportunities):
                 gap_type = next((opp.get('content_type', 'alternative') for opp in content_opportunities if opp.get('type') == 'content_type_gap'), 'alternative')
                 next_steps.append({
                    "step": "2b", # Use a letter step
                    "title": f"Develop {gap_type} content",
                    "description": f"Create content in alternative formats like {gap_type} (e.g., interactive tools, videos, infographics) to fill gaps in the competitive landscape."
                })

            # Step for intent targeting if identified
            if any(rec.get('type') == 'intent_targeting' for rec in insights.get('content_strategy', {}).get('recommendations', [])):
                 intent_rec = next((rec.get('title') for rec in insights.get('content_strategy', {}).get('recommendations', []) if rec.get('type') == 'intent_targeting'), None)
                 if intent_rec:
                     next_steps.append({
                        "step": "2c", # Use a letter step
                        "title": intent_rec, # Use the title from the recommendation
                        "description": next((rec.get('description', '') for rec in insights.get('content_strategy', {}).get('recommendations', []) if rec.get('type') == 'intent_targeting'), '')
                    })


            # Step for optimizing for specific SERP features if identified
            if serp_insights_list:
                top_feature = serp_insights_list[0].get('feature', 'SERP features').replace('_', ' ')
                next_steps.append({
                    "step": 3,
                    "title": f"Optimize for {top_feature}",
                    "description": f"Implement structured data, content formatting (like headings and lists), or create specific media (like video) to target the identified {top_feature} in search results."
                })
            else:
                 # If no specific features were found, add a general SERP optimization step
                 next_steps.append({
                    "step": 3,
                    "title": "General SERP optimization",
                    "description": "Focus on standard on-page SEO best practices, including optimizing titles, meta descriptions, headings, and content quality to improve visibility in search results."
                })


            next_steps.append({
                "step": 4,
                "title": "Develop content plan & create content",
                "description": "Create a content calendar based on your prioritized keywords, identified opportunities, and recommended content types/formats. Develop high-quality content that addresses user intent and stands out from competitors."
            })


            next_steps.append({
                "step": 5,
                "title": "Build authority & promote content",
                "description": "Acquire relevant backlinks to your content and promote it across relevant channels (social media, email, paid ads) to improve its ranking and visibility."
            })


            next_steps.append({
                "step": 6,
                "title": "Monitor performance",
                "description": "Track keyword rankings, organic traffic, user engagement metrics, and conversions for your targeted keywords and content. Use this data to refine your strategy."
            })

            # Sort steps by step number
            next_steps.sort(key=lambda x: str(x.get('step', 99))) # Sorts 1, 2, 2a, 2b, 3...


            return next_steps
        except Exception as e:
            logger.error(f"Error generating next steps: {str(e)}")
            return [
                {
                    "step": 1,
                    "title": "Review analysis results",
                    "description": "Carefully examine the keyword data, competitive landscape, and identified opportunities."
                },
                {
                    "step": 2,
                    "title": "Develop an action plan",
                    "description": "Based on the insights, create a specific plan for content creation, optimization, and promotion."
                }
            ]

    def _get_current_date(self):
        """
        Get the current date in YYYY-MM-DD format

        Returns:
            str: Current date
        """
        return datetime.now().strftime("%Y-%m-%d")

