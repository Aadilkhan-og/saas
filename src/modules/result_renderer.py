# Filename: src/modules/result_renderer.py
import logging
import json
from datetime import datetime
import numpy as np # Needed for potential calculations, like averages if not in summary

logger = logging.getLogger("keyword_research.result_renderer")

class ResultRenderer:
    """
    Renders the final results in a structured format.
    Structures the enhanced data and AI insights for frontend display.
    """

    def __init__(self):
        pass

    # Updated signature to accept full serp_data, competitor_data, insights, etc.
    def render_results(self, input_data, keyword_analysis, serp_data, competitor_data, insights, raw_serp_data, raw_competitor_data):
        """
        Render the final results from all collected and analyzed data.

        Args:
            input_data (dict): Processed user input.
            keyword_analysis (dict): Processed keyword data (intent, scores, clusters, questions).
            serp_data (dict): Comprehensive SerpAPI data.
            competitor_data (dict): Detailed Competitor content analysis.
            insights (dict): Generated insights (AI and rule-based).
            raw_serp_data (dict): Same as serp_data, passed for clarity in original structure.
            raw_competitor_data (dict): Same as competitor_data, passed for clarity in original structure.

        Returns:
            dict: Structured results for presentation, including detailed data.
        """
        logger.info("Rendering final results, including detailed data and AI insights.")

        try:
            # Prepare the main result structure
            # Populate sections directly from the processed/insight data where possible
            results = {
                "summary": { # Executive Summary (AI generated)
                    "title": f"Keyword Research: {input_data.get('seed_keyword', 'Unknown')}",
                    "date": self._get_current_date(),
                    "keyword_count": len(input_data.get('keywords', [])),
                    # Use the AI-generated summary directly
                    "insight_summary": insights.get('summary', {}).get('overall_summary', 'Summary not available.'),
                    # Use AI-generated key opportunities highlight
                    "top_opportunities": insights.get('summary', {}).get('key_opportunities_highlight', []),
                },
                "keywords": { # Keyword Analysis section
                    # 'analyzed' keywords with detailed scores (from keyword_analysis)
                    "analyzed": self._format_keywords_with_scores(input_data.get('keywords', []), keyword_analysis),
                    # Question keywords (from keyword_analysis)
                    "questions": keyword_analysis.get('question_keywords', []),
                    # Keyword Recommendations (from insights)
                    "opportunities": insights.get('keyword_recommendations', []), # Renamed from opportunities in old structure
                },
                "intent_analysis": { # Intent & Clustering
                    # Intent distribution (from insights/keyword_analysis)
                    "distribution": insights.get('intent_distribution', keyword_analysis.get('intent_distribution', {})),
                     # Keyword Clusters (from insights/keyword_analysis)
                    "clusters": insights.get('topic_clusters', keyword_analysis.get('clusters', [])),
                },
                "serp_insights": { # Competitive & SERP Insights
                     # AI Competitive Landscape Summary - NEW
                    "competitive_landscape_summary": insights.get('competitive_landscape_summary', {}),
                    # Top Competitors (from basic analysis fallback or SerpAPI data)
                    "top_competitors": self._extract_top_competitors(serp_data), # Use serp_data for domain distribution
                    # Aggregated Competitor Content Summary (from competitor_data.summary)
                    "competitor_content_aggregated_summary": competitor_data.get('summary', {}),
                    # Actionable SERP feature insights (from insights)
                    "features": insights.get('serp_feature_insights', []), # Renamed from 'features' in old structure
                },
                 # AI Content Blueprints - NEW
                "content_blueprints": insights.get('content_blueprints', {}),


                # Detailed SERP data (using raw_serp_data which is the comprehensive SerpAPI data)
                # Need to format this for display, often just returning the relevant part of the raw data is easiest for frontend
                "detailed_serp_results": {
                     # Frontend will need to pick a keyword (e.g., seed) to show detailed results for
                     # Provide the full raw data structure for a target keyword (frontend picks which one)
                     # Or provide the full raw serp_data and let frontend handle it
                    "full_serp_data": serp_data, # Pass the full comprehensive data
                    # Optionally, pre-select data for the seed keyword for easier frontend access
                    "seed_keyword_serp_data": serp_data.get('serp_data', {}).get(input_data.get('seed_keyword'), []),
                    "seed_keyword_features": serp_data.get('features', {}).get(input_data.get('seed_keyword'), []),
                    "seed_keyword_paa_questions": serp_data.get('paa_questions', {}).get(input_data.get('seed_keyword'), []),
                    "seed_keyword_related_searches": serp_data.get('related_searches', {}).get(input_data.get('seed_keyword'), []),
                     "seed_keyword_featured_snippet": serp_data.get('featured_snippet'), # SerpAPI might return one overall FS or per keyword
                     # If SerpAPI returns FS per keyword in serp_data['serp_data'][kw]['featured_snippet'], adjust here
                     # Assuming for now it might be a top-level field or needs searching
                     "overall_featured_snippet": serp_data.get('featured_snippet'), # Keep top-level FS if present

                },


                 # Detailed Competitor content analysis (using raw_competitor_data)
                "detailed_content_analysis": self._format_detailed_content_analysis(competitor_data), # Use competitor_data


                "content_strategy": { # Content Strategy Recommendations
                    # Use the AI/rule-based opportunities from insights
                    "opportunities": [opp for opp in insights.get('content_opportunities', []) if opp.get('type') not in ['intent_targeting', 'content_type_gap']], # Exclude types now covered by recommendations
                     # Use the AI/rule-based recommendations from insights
                    "recommendations": [rec for rec in insights.get('content_opportunities', []) if rec.get('type') in ['intent_targeting', 'content_type_gap']] + insights.get('keyword_recommendations', []), # Combine relevant types from opportunities and keyword recs
                },


                "next_steps": self._generate_next_steps(keyword_analysis, insights), # Generate next steps based on insights
            }

             # --- Clean up / Ensure consistency ---
             # Ensure lists in the final output are always lists
            for section in ["keywords", "serp_insights", "content_strategy"]:
                 for key in ["analyzed", "questions", "opportunities", "top_competitors", "features", "opportunities", "recommendations"]:
                      if key in results.get(section, {}) and not isinstance(results[section][key], list):
                           results[section][key] = []

            for key in ["clusters"]:
                 if key in results.get("intent_analysis", {}) and not isinstance(results["intent_analysis"][key], list):
                      results["intent_analysis"][key] = []

            if not isinstance(results.get("content_blueprints"), dict):
                 results["content_blueprints"] = {}


            return results

        except Exception as e:
            logger.error(f"Error rendering results: {str(e)}", exc_info=True)
            # Return minimal results on error
            return {
                "summary": {
                    "title": f"Keyword Research (Partial): {input_data.get('seed_keyword', 'Unknown')}",
                    "date": self._get_current_date(),
                    "keyword_count": len(input_data.get('keywords', [])),
                    "insight_summary": "An error occurred while rendering results. Some data may be missing.",
                    "top_opportunities": []
                },
                "error": str(e),
                # Include potentially available partial data for debugging
                "partial_input_data": input_data,
                "partial_keyword_analysis": keyword_analysis,
                "partial_serp_data": serp_data,
                "partial_competitor_data": competitor_data,
                "partial_insights": insights,
            }

    def _format_keywords_with_scores(self, keywords, keyword_analysis):
        """
        Format keywords with their detailed analysis data (intent, scores).
        Uses the enhanced scores from keyword_analysis.

        Args:
            keywords (list): List of analyzed keywords (original input).
            keyword_analysis (dict): Processed keyword data including enhanced scores.

        Returns:
            list: Formatted keyword data with detailed scores.
        """
        try:
            formatted_keywords = []
            intent_classification = keyword_analysis.get('intent_classification', {})
            keyword_scores = keyword_analysis.get('keyword_scores', {}) # Use the enhanced scores

            # Iterate through the keywords that were actually scored/classified in keyword_analysis
            # This ensures we only include keywords for which we have analysis data
            analyzed_kws_in_data = set(intent_classification.keys()).union(set(keyword_scores.keys()))


            for keyword in keywords: # Iterate through original input keywords
                 if keyword in analyzed_kws_in_data: # Only include if analysis data exists
                    score_data = keyword_scores.get(keyword, {"search_volume": None, "keyword_difficulty": None, "difficulty": 50, "opportunity": 50, "score": 50}) # Provide default scores if missing
                    keyword_data = {
                        "keyword": keyword,
                        "intent": intent_classification.get(keyword, "unknown"),
                        "scores": {
                            "search_volume": score_data.get("search_volume"), # Include raw SV
                            "keyword_difficulty": score_data.get("keyword_difficulty"), # Include raw KD
                            "difficulty": score_data.get("difficulty", 50), # Calculated difficulty
                            "opportunity": score_data.get("opportunity", 50), # Calculated opportunity
                            "score": score_data.get("score", 50) # Calculated overall score
                         }
                    }
                    formatted_keywords.append(keyword_data)

            # Sort by overall score if available
            formatted_keywords.sort(
                key=lambda x: x.get('scores', {}).get('score', 0),
                reverse=True
            )

            return formatted_keywords
        except Exception as e:
            logger.error(f"Error formatting keywords with scores: {str(e)}")
            # Fallback to basic formatting if error occurs
            return [{"keyword": kw, "intent": keyword_analysis.get('intent_classification', {}).get(kw, 'unknown'), "scores": {"search_volume": None, "keyword_difficulty": None, "difficulty": 50, "opportunity": 50, "score": 50}} for kw in keywords]


    def _extract_top_opportunities(self, insights):
        """
        Extract top opportunities from insights - This method might become redundant
        as the frontend now uses insights.summary.key_opportunities_highlight
        and insights.content_opportunities + insights.keyword_recommendations directly.
        Retaining for now but noted.
        """
        logger.warning("ResultRenderer._extract_top_opportunities is used but might be redundant.")
        try:
            # This method's logic is similar to how the frontend renders the main opportunities section.
            # The AI summary's key_opportunities_highlight is likely better.
            # We can return a combined list for potential use, or rely on the frontend.
            # Let's return a simple list combining a few key AI highlights and top keyword recommendations.
            all_opportunities = []

            # Add AI summary highlights
            all_opportunities.extend(insights.get('summary', {}).get('key_opportunities_highlight', []))

            # Add a few top keyword recommendations (if not already covered by summary highlight)
            keyword_recs = insights.get('keyword_recommendations', [])
            for rec in keyword_recs[:3]: # Take top 3 keyword recs
                 rec_text = rec.get('keyword', '') # Use keyword as identifier
                 if rec_text and rec_text not in all_opportunities: # Avoid simple string duplicates
                      all_opportunities.append(f"Target keyword '{rec_text}' ({rec.get('type', 'recommendation')})")

            return all_opportunities[:5] # Limit the overall list

        except Exception as e:
            logger.error(f"Error extracting top opportunities (redundant method): {str(e)}")
            return []


    def _extract_top_competitors(self, serp_data):
        """
        Extract top competitors from SerpAPI data (domain distribution).

        Args:
            serp_data (dict): Comprehensive SerpAPI data.

        Returns:
            list: Top competitors as list of {domain: str, percentage: int}.
        """
        logger.info("Extracting top competitors from SerpAPI data.")
        competitors = []
        if serp_data and "serp_data" in serp_data:
             domain_counts = {}
             total_organic_results = 0
             for keyword, results in serp_data["serp_data"].items():
                  if isinstance(results, list):
                      for result in results:
                          if "link" in result: # Use 'link' from SerpAPI structure
                              domain = self._extract_domain(result["link"])
                              domain_counts[domain] = domain_counts.get(domain, 0) + 1
                              total_organic_results += 1

             # Calculate domain distribution (percentage of total organic results)
             for domain, count in domain_counts.items():
                 # Include domains that appear more than once, or always include if total results is small
                 if count > 1 or total_organic_results <= 10: # Be less strict for smaller result sets
                     percentage = (count / total_organic_results) * 100 if total_organic_results > 0 else 0
                     competitors.append({
                         "domain": domain,
                         "percentage": round(percentage)
                     })

             # Sort by percentage
             competitors.sort(key=lambda x: x.get('percentage', 0), reverse=True)

             # Return top 10 (or adjust number)
             return competitors[:10]
        else:
             logger.warning("No SerpAPI data available to extract top competitors.")
             return []


    # --- Detailed Content Analysis Formatting ---
    def _format_detailed_content_analysis(self, competitor_data):
        """
        Formats the detailed competitor content analysis data for the report.
        Uses the detailed analysis from competitor_data.content_analysis.

        Args:
            competitor_data (dict): Full, detailed Competitor content analysis data.

        Returns:
            list: List of formatted analysis results per URL.
        """
        formatted_list = []
        # competitor_data['content_analysis'] holds the dict {url: {analysis_data}}
        content_analysis_by_url = competitor_data.get('content_analysis', {})
        analyzed_urls = competitor_data.get('analyzed_urls', []) # Use the list of successfully analyzed URLs

        # Iterate through the URLs that were actually analyzed
        for url in analyzed_urls:
            analysis_data = content_analysis_by_url.get(url)
            if analysis_data: # Should exist if in analyzed_urls, but double-check
                formatted_list.append({
                    "url": url,
                    "analysis": analysis_data # Include the full detailed analysis data for this URL
                })
        return formatted_list


    def _generate_content_recommendations(self, keyword_analysis, insights, competitor_data):
        """
        Generate content recommendations based on the analysis.
        This method might become redundant as insights.content_opportunities and
        insights.keyword_recommendations are now the primary sources and are passed directly.
        Retaining for now but noted.
        """
        logger.warning("ResultRenderer._generate_content_recommendations is used but might be redundant.")
        # This method's logic is largely superseded by the AI/enhanced rule-based
        # opportunities and recommendations generated in InsightGenerator.
        # Returning a combined list from those sources for compatibility.
        try:
            # Combine relevant lists from insights
            recommendations = []
            # Add AI/basic content opportunities
            recommendations.extend(insights.get('content_opportunities', []))
            # Add AI/enhanced keyword recommendations
            recommendations.extend(insights.get('keyword_recommendations', []))

            # Remove potential duplicates if items can appear in both lists
            unique_recommendations = {}
            for rec in recommendations:
                # Create a key based on type and title/keyword for uniqueness check
                key = (rec.get('type'), rec.get('title') or rec.get('keyword'))
                if key not in unique_recommendations:
                    unique_recommendations[key] = rec

            return list(unique_recommendations.values())

        except Exception as e:
            logger.error(f"Error generating content recommendations (redundant method): {str(e)}")
            return []


    def _generate_next_steps(self, keyword_analysis, insights):
        """
        Generate recommended next steps based on the analysis and insights.
        Updated to use the new insights structure.

        Args:
            keyword_analysis (dict): Processed keyword data.
            insights (dict): Generated insights (AI and rule-based).

        Returns:
            list: Recommended next steps.
        """
        logger.info("Generating recommended next steps.")
        try:
            next_steps = []
            step_counter = 1 # For numbering steps

            # Use the AI summary highlights if available
            ai_summary_highlights = insights.get('summary', {}).get('key_opportunities_highlight', [])
            if ai_summary_highlights:
                next_steps.append({
                    "step": step_counter,
                    "title": "Review Key AI-Identified Opportunities",
                    "description": f"Focus on the top opportunities highlighted in the Executive Summary, such as: {'; '.join(ai_summary_highlights)}. Prioritize those aligning with your business goals."
                })
                step_counter += 1
            else:
                 # If no AI highlights, provide a basic step
                 next_steps.append({
                    "step": step_counter,
                    "title": "Review Analysis Findings",
                    "description": "Carefully examine the keyword data, competitive landscape summary, and identified opportunities to understand the key findings for your niche."
                })
                 step_counter += 1


            # Step based on Content Blueprints availability
            if insights.get('content_blueprints'):
                 blueprint_keywords = list(insights['content_blueprints'].keys())
                 next_steps.append({
                     "step": step_counter,
                     "title": f"Review & Utilize Content Blueprint(s)",
                     "description": f"Examine the AI-generated content blueprint(s) for '{'; '.join(blueprint_keywords)}'. Use the suggested outlines, themes, word counts, and unique angle ideas as a starting point for content creation."
                 })
                 step_counter += 1
            else:
                 # If no blueprints, suggest manual outline creation based on detailed analysis
                 next_steps.append({
                     "step": step_counter,
                     "title": "Analyze Competitor Content Details & Plan Structure",
                     "description": "Review the 'Detailed Competitor Content Analysis' to understand the structure, themes, and angles used by top-ranking pages. Manually create a content outline based on these insights."
                 })
                 step_counter += 1


            # Step based on Keyword Recommendations (High Score, Question, Related)
            if insights.get('keyword_recommendations'):
                 high_score_count = len([r for r in insights['keyword_recommendations'] if r.get('type') == 'high_score'])
                 question_count = len([r for r in insights['keyword_recommendations'] if r.get('type') == 'question'])
                 related_count = len([r for r in insights['keyword_recommendations'] if r.get('type') == 'related_search'])

                 recommendation_summary_parts = []
                 if high_score_count > 0: recommendation_summary_parts.append(f"{high_score_count} high-scoring keywords")
                 if question_count > 0: recommendation_summary_parts.append(f"{question_count} question keywords")
                 if related_count > 0: recommendation_summary_parts.append(f"{related_count} related search terms")

                 if recommendation_summary_parts:
                      next_steps.append({
                         "step": step_counter,
                         "title": "Prioritize & Select Target Keywords",
                         "description": f"Review the 'Recommended Keywords' list. Select primary and secondary target keywords from the {', '.join(recommendation_summary_parts)} based on your resources and relevance."
                      })
                      step_counter += 1


            # Step based on SERP Feature Insights
            if insights.get('serp_feature_insights'):
                 top_feature_insights = insights['serp_feature_insights'][:2] # Mention top 2 features
                 if top_feature_insights:
                      feature_titles = [f.get('feature', '').replace('_', ' ') for f in top_feature_insights]
                      next_steps.append({
                         "step": step_counter,
                         "title": f"Optimize for Key SERP Features ({', '.join(feature_titles)})",
                         "description": f"Refer to the 'SERP Features Observed & Optimization' section for actionable strategies to target features like {', '.join(feature_titles)} in search results (e.g., structuring content for snippets, creating videos)."
                      })
                      step_counter += 1


            # Step based on Intent-Matched Recommendations (from content_opportunities in insights)
            intent_recs = [opp for opp in insights.get('content_opportunities', []) if opp.get('type') in ['intent_gap', 'content_type_gap']]
            if intent_recs:
                 intent_titles = [rec.get('title') for rec in intent_recs[:2]]
                 next_steps.append({
                     "step": step_counter,
                     "title": f"Align Content Formats with User Intent ({', '.join(intent_titles)})",
                     "description": f"Consider the recommended content formats and strategies based on user intent and competitive gaps. Ensure your planned content aligns with what users are looking for."
                 })
                 step_counter += 1


            # Standard content creation and promotion steps
            next_steps.append({
                "step": step_counter,
                "title": "Develop Content Plan & Create Content",
                "description": "Create a content calendar based on your prioritized keywords, selected blueprint(s), and identified opportunities. Develop high-quality content that addresses user intent and stands out from competitors."
            })
            step_counter += 1

            next_steps.append({
                "step": step_counter,
                "title": "Build Authority & Promote Content",
                "description": "Acquire relevant backlinks to your new content and promote it across relevant channels (social media, email, paid ads) to improve its ranking and visibility."
            })
            step_counter += 1

            next_steps.append({
                "step": step_counter,
                "title": "Monitor Performance & Refine Strategy",
                "description": "Track keyword rankings, organic traffic, user engagement metrics, and conversions for your targeted keywords and content. Use this data to refine your content and SEO strategy."
            })
            step_counter += 1


            # Sort steps by step number (handles integer and string steps like '2a')
            next_steps.sort(key=lambda x: str(x.get('step', 99)))


            return next_steps
        except Exception as e:
            logger.error(f"Error generating next steps: {str(e)}")
            # Fallback to basic next steps on error
            return [
                {
                    "step": 1,
                    "title": "Review Analysis Results",
                    "description": "Carefully examine the keyword data, competitive landscape, and identified opportunities."
                },
                {
                    "step": 2,
                    "title": "Develop an Action Plan",
                    "description": "Based on the insights, create a specific plan for content creation, optimization, and promotion."
                }
            ]


    # --- Helper methods from original code, keep and adapt if needed ---

    def _get_current_date(self):
        """Get the current date in YYYY-MM-DD format."""
        return datetime.now().strftime("%Y-%m-%d")

    def _extract_domain(self, url):
        """Extract domain from a URL - Keep existing method."""
        try:
            from urllib.parse import urlparse

            if not url:
                return ""

            url = str(url)
            url = url.replace('"https":', 'https://').replace('"http":', 'http://')
            url = url.replace('"', '')

            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            if domain.startswith('www.'):
                domain = domain[4:]
             # Handle googleusercontent.com for YouTube links
            if "googleusercontent.com" in domain and "youtube.com" in url:
                 return "youtube.com"

            return domain
        except Exception:
            return url