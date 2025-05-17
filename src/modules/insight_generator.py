# Filename: src/modules/insight_generator.py
import asyncio
import os
import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator # For structured output parsing (more robust)
from typing import List, Optional, Dict, Any
import numpy as np # Import numpy

logger = logging.getLogger("keyword_research.insight_generator")

# Define Pydantic models for structured output parsing from LLM (Optional but Recommended)
# This makes parsing more reliable if the LLM can follow the schema instructions.

async def invoke_with_retry(chain, payload, retries=5, delay=5):
    for attempt in range(retries):
        try:
            # Try async invoke if available, else fallback to sync
            if asyncio.iscoroutinefunction(chain.invoke):
                return await chain.invoke(payload)
            else:
                return chain.invoke(payload)
        except Exception as e:
            # Check for 429 or ResourceExhausted
            if hasattr(e, "status_code") and e.status_code == 429:
                logger.warning(f"Rate limit hit (429). Retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                await asyncio.sleep(delay)
            elif "ResourceExhausted" in str(e):
                logger.warning(f"Resource exhausted. Retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                await asyncio.sleep(delay)
            else:
                logger.warning(f"Error in invoke_with_retry: {e}. Retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                await asyncio.sleep(delay)
    raise RuntimeError(f"invoke_with_retry failed after {retries} attempts.")

class ContentBlueprintSection(BaseModel):
    """Represents a section in a content outline blueprint."""
    level: int = Field(..., description="Heading level (e.g., 1 for H1, 2 for H2, 3 for H3).")
    title: str = Field(..., description="Proposed title for the section.")
    themes_to_cover: List[str] = Field(default_factory=list, description="Key themes or sub-topics to cover in this section.")
    notes: Optional[str] = Field(None, description="Optional notes or specific points for this section.")
    sub_sections: List["ContentBlueprintSection"] = Field(default_factory=list) # Nested sections

# Need to handle forward reference for recursive model
ContentBlueprintSection.update_forward_refs()

class ContentBlueprint(BaseModel):
    """Represents a suggested content blueprint for a keyword."""
    target_keyword: str = Field(..., description="The primary keyword the blueprint is for.")
    recommended_content_type: str = Field(..., description="Recommended content format (e.g., Blog Post, Guide, Comparison).")
    recommended_word_count_range: str = Field(..., description="Suggested word count range (e.g., '1500-2000 words').")
    key_themes_overall: List[str] = Field(default_factory=list, description="Overall key themes identified for the topic.")
    suggested_outline: List[ContentBlueprintSection] = Field(default_factory=list, description="The structured content outline.")
    unique_angle_ideas: List[str] = Field(default_factory=list, description="Ideas for differentiating the content.")
    target_paa_questions: List[str] = Field(default_factory=list, description="Relevant People Also Ask questions to address.")


class CompetitiveStrategySummary(BaseModel):
    """Summarizes the competitive content strategy."""
    dominant_content_types: List[str] = Field(default_factory=list, description="Most common content formats among top competitors.")
    average_word_count: str = Field(..., description="Average word count of ranking content.")
    common_themes_and_angles: List[str] = Field(default_factory=list, description="Common themes, topics, and angles used by top competitors.")
    common_structure_patterns: List[str] = Field(default_factory=list, description="Common heading structures or content organization patterns.")
    common_media_types: List[str] = Field(default_factory=list, description="Most common media types used.")
    winning_formula_summary: str = Field(..., description="A brief summary of the apparent 'winning formula' in the SERP.")


class ActionableSerpFeatureInsight(BaseModel):
    """Provides actionable advice for a specific SERP feature."""
    feature: str = Field(..., description="The SERP feature (e.g., 'featured_snippet', 'people_also_ask').")
    description: str = Field(..., description="Brief description of the feature's presence.")
    strategy: str = Field(..., description="Specific, actionable steps to optimize for this feature.")

class SerpFeatureInsightList(BaseModel):
    __root__: List[ActionableSerpFeatureInsight]

class IntentMatchedRecommendation(BaseModel):
    """Recommendation for content format based on intent and competition."""
    intent: str = Field(..., description="The primary intent analyzed.")
    recommendation_type: str = Field(..., description="Type of recommendation (e.g., 'content_format', 'intent_gap').")
    title: str = Field(..., description="Title of the recommendation.")
    description: str = Field(..., description="Detailed explanation of the recommendation.")


class InsightSummary(BaseModel):
     """Overall summary of the keyword research analysis."""
     overall_summary: str = Field(..., description="A concise summary highlighting key findings, opportunities, and recommendations.")
     key_opportunities_highlight: List[str] = Field(default_factory=list, description="A few top opportunities highlighted from the analysis.")


class InsightGenerator:
    """
    Generates insights from the collected keyword and competitor data using LLM.
import asyncio


    """

    def __init__(self):
        try:
            self.api_key = os.getenv("GOOGLE_API_KEY")
            self.model = "gemini-2.0-flash"
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=0.7
            )
            logger.info("ChatGoogleGenerativeAI (Gemini) initialized for InsightGenerator")
            self.llm_available = True
        except Exception as e:
            logger.error(f"Error initializing ChatGoogleGenerativeAI in InsightGenerator: {str(e)}")
            self.llm = None
            self.llm_available = False

        # Initialize output parsers for structured output (optional, but good)
        if self.llm_available:
             self.blueprint_parser = JsonOutputParser(pydantic_object=ContentBlueprint)
             self.competitive_summary_parser = JsonOutputParser(pydantic_object=CompetitiveStrategySummary)
             # Adjust parser for list output if needed, or handle list parsing manually
             self.serp_feature_parser = JsonOutputParser(pydantic_object=SerpFeatureInsightList)
             self.intent_recommendation_parser = JsonOutputParser(pydantic_object=List[IntentMatchedRecommendation])
             self.insight_summary_parser = JsonOutputParser(pydantic_object=InsightSummary)
        else:
             self.blueprint_parser = None
             self.competitive_summary_parser = None
             self.serp_feature_parser = None
             self.intent_recommendation_parser = None
             self.insight_summary_parser = None


    async def generate_insights(self, input_data, keyword_analysis, serp_data, competitor_data):
        """
        Generate insights from the collected data using LLM.

        Args:
            input_data (dict): Processed user input (includes seed keyword)
            keyword_analysis (dict): Processed keyword data (intent, scores, clusters, questions)
            serp_data (dict): Comprehensive SerpAPI data for keywords (includes SV, KD, features, results)
            competitor_data (dict): Detailed Competitor content analysis

        Returns:
            dict: Generated insights
        """
        logger.info("Generating insights from the collected data")

        # Initialize insights structure with new fields
        insights = {
            "content_opportunities": [], # Opportunities based on analysis (can be AI or rule-based)
            "serp_feature_insights": [], # Actionable SERP feature insights (AI)
            "competitive_landscape_summary": {}, # Summary of competitive content (AI)
            "keyword_recommendations": [], # Keyword recommendations (rule-based + scoring)
            "topic_clusters": [], # Keyword clusters (from processor)
            "intent_distribution": {}, # Intent distribution (from processor)
            "content_blueprints": {}, # Generated content blueprints per target keyword (AI) - NEW
            "summary": {} # Overall AI-generated summary - NEW (was string)
        }

        # Extract data needed for insight generation
        keywords = input_data.get("keywords", [])
        seed_keyword = input_data.get("seed_keyword", "")
        intent_classification = keyword_analysis.get("intent_classification", {})
        keyword_scores = keyword_analysis.get("keyword_scores", {})
        question_keywords = keyword_analysis.get("question_keywords", [])
        topic_clusters = keyword_analysis.get("clusters", []) # Use clusters from processor

        # Ensure SerpAPI data is available
        serp_api_serp_data = serp_data.get("serp_data", {}) # Detailed organic results per keyword
        serp_api_features = serp_data.get("features", {}) # Detected features per keyword
        serp_api_paa = serp_data.get("paa_questions", {}) # PAA per keyword
        serp_api_related = serp_data.get("related_searches", {}) # Related searches per keyword
        serp_api_sv = serp_data.get("search_volume", {}) # SV per keyword
        serp_api_kd = serp_data.get("keyword_difficulty", {}) # KD per keyword

        # Ensure detailed competitor data is available
        analyzed_urls = competitor_data.get("analyzed_urls", [])
        detailed_content_analysis = competitor_data.get("content_analysis", {})
        competitor_summary_data = competitor_data.get("summary", {}) # Aggregated content summary data


        # Use LLM for sophisticated insights if available
        if self.llm_available:
            try:
                # 1. Generate Overall Summary (AI)
                logger.info("Generating overall insights summary using LLM.")
                insights["summary"] = await self._generate_summary_with_llm(
                    input_data, keyword_analysis, serp_data, competitor_data, insights # Pass all data for context
                )
                logger.info("Overall summary generated.")

                # 2. Analyze Competitive Landscape & Summarize Winning Formula (AI)
                logger.info("Analyzing competitive landscape and summarizing winning formula.")
                insights["competitive_landscape_summary"] = await self._summarize_competitive_content_strategy(
                    analyzed_urls, detailed_content_analysis, competitor_summary_data, serp_api_features
                )
                logger.info("Competitive landscape summary generated.")


                # 3. Generate Actionable SERP Feature Insights (AI)
                logger.info("Generating actionable SERP feature insights.")
                insights["serp_feature_insights"] = await self._generate_serp_feature_insights_ai(
                    serp_api_features, keywords, keyword_scores # Pass relevant data
                )
                # Ensure the result is a list, even if LLM returns empty or error
                if not isinstance(insights["serp_feature_insights"], list):
                     insights["serp_feature_insights"] = self._generate_serp_feature_insights_basic(serp_data) # Fallback if AI fails list format

                logger.info(f"Generated {len(insights['serp_feature_insights'])} SERP feature insights.")


                # 4. Generate Intent-Matched Content Format Recommendations (AI)
                logger.info("Generating intent-matched content recommendations.")
                ai_intent_recs = await self._generate_intent_matched_recommendations_ai(
                     intent_classification, competitor_summary_data, serp_api_features
                )
                 # Ensure the result is a list
                if isinstance(ai_intent_recs, list):
                     insights["content_opportunities"].extend(ai_intent_recs)
                else:
                     logger.warning("AI intent recommendations did not return a list.")
                     # Fallback for intent recommendations if AI fails list format
                     insights["content_opportunities"].extend(self._generate_content_opportunities_basic(keyword_analysis, serp_data, competitor_data))


                logger.info(f"Generated {len(insights['content_opportunities'])} intent-matched recommendations (including fallbacks).")


                # 5. Generate Content Blueprints (AI)
                # Generate a blueprint for the seed keyword and maybe a couple of high-opportunity keywords
                target_keywords_for_blueprint = [seed_keyword]
                # Add 1-2 high opportunity keywords to generate blueprints for
                high_opp_keywords = sorted(keyword_scores.items(), key=lambda item: item[1].get("score", 0), reverse=True)[:5] # Look at top 5 scored
                keywords_added = 0
                for kw, score_data in high_opp_keywords:
                     # Check if keyword is one of the original inputs and has a good score
                     if kw != seed_keyword and kw in keywords and score_data.get("score", 0) > 60: # High score & was an input keyword
                          if keywords_added < 2: # Limit to 2 additional blueprints
                               target_keywords_for_blueprint.append(kw)
                               keywords_added += 1


                logger.info(f"Generating content blueprints for keywords: {target_keywords_for_blueprint}")
                for target_kw in target_keywords_for_blueprint:
                     # *** FIX: Pass serp_data to _generate_content_blueprint ***
                     blueprint = await self._generate_content_blueprint(
                          target_kw,
                          serp_api_serp_data.get(target_kw, []), # Organic results for target kw
                          detailed_content_analysis, # All detailed competitor analysis
                          keyword_analysis, # Pass full keyword analysis for context (questions, clusters)
                          insights.get("competitive_landscape_summary", {}), # Pass competitive summary
                          serp_data # *** FIX: Pass serp_data here ***
                          )
                     if blueprint:
                          insights["content_blueprints"][target_kw] = blueprint
                          logger.info(f"Generated blueprint for '{target_kw}'.")
                     else:
                          logger.warning(f"Failed to generate blueprint for '{target_kw}'.")


                # 6. Generate Keyword Recommendations (Rule-based + Scoring + maybe AI refinement)
                # Re-use and enhance the existing logic to leverage SV/KD and potentially add AI refinement
                insights["keyword_recommendations"] = self._generate_keyword_recommendations_enhanced(
                    keyword_analysis, serp_data # Pass data including SV/KD, PAA, Related Searches
                )
                logger.info(f"Generated {len(insights['keyword_recommendations'])} keyword recommendations.")


            except Exception as e:
                logger.error(f"Error during main LLM insight generation block: {str(e)}", exc_info=True)
                # Fallback or partial insights if LLM generation fails
                # Generate basic summary as fallback
                insights["summary"] = {"overall_summary": self._generate_static_summary(keyword_analysis, serp_data, competitor_data, insights)}
                # Fallback for other sections if they failed in the AI block
                if not insights.get("competitive_landscape_summary"):
                     insights["competitive_landscape_summary"] = self._analyze_competitive_landscape_basic(serp_data, competitor_data)
                # SERP features and Intent recs already have fallbacks in their calls
                insights["keyword_recommendations"] = self._generate_keyword_recommendations_enhanced(keyword_analysis, serp_data) # Ensure rule-based is run
                # Content blueprints will be empty if AI fails


        else:
            logger.warning("LLM not available. Generating basic insights only.")
            # Fallback to more basic, rule-based insights if LLM is not available
            insights["summary"] = {"overall_summary": self._generate_static_summary(keyword_analysis, serp_data, competitor_data, insights)}
            insights["serp_feature_insights"] = self._generate_serp_feature_insights_basic(serp_data)
            insights["competitive_landscape_summary"] = self._analyze_competitive_landscape_basic(serp_data, competitor_data)
            insights["keyword_recommendations"] = self._generate_keyword_recommendations_enhanced(keyword_analysis, serp_data) # Still use enhanced rule-based
            insights["content_opportunities"] = self._generate_content_opportunities_basic(keyword_analysis, serp_data, competitor_data) # Basic content opps
            insights["content_blueprints"] = {} # Blueprints require LLM


        # Ensure clusters and intent distribution from processor are included
        insights["topic_clusters"] = topic_clusters
        insights["intent_distribution"] = self._calculate_intent_distribution(keyword_analysis)


        # Ensure lists are lists, even if fallbacks/errors occurred
        insights["content_opportunities"] = insights.get("content_opportunities", [])
        insights["serp_feature_insights"] = insights.get("serp_feature_insights", [])
        insights["keyword_recommendations"] = insights.get("keyword_recommendations", [])
        insights["topic_clusters"] = insights.get("topic_clusters", [])


        return insights


    # --- AI-POWERED INSIGHT GENERATION METHODS ---

    async def _generate_summary_with_llm(self, input_data, keyword_analysis, serp_data, competitor_data, current_insights):
         """Generate an overall summary using the LLM."""
         if not self.llm_available or not self.insight_summary_parser:
              return {"overall_summary": self._generate_static_summary(keyword_analysis, serp_data, competitor_data, current_insights)}

         try:
             # Prepare context for the LLM
             context = {
                 "seed_keyword": input_data.get("seed_keyword"),
                 "total_keywords_analyzed_count": len(keyword_analysis.get('intent_classification', {})),
                 "intent_distribution": self._calculate_intent_distribution(keyword_analysis),
                 "top_keywords_by_score": sorted(keyword_analysis.get("keyword_scores", {}).items(), key=lambda item: item[1].get("score", 0), reverse=True)[:5],
                 "question_keywords_count": len(keyword_analysis.get("question_keywords", [])),
                 "avg_word_count": competitor_data.get("summary", {}).get("avg_word_count"),
                 "avg_readability": competitor_data.get("summary", {}).get("avg_readability_score"),
                 "most_common_content_type": competitor_data.get("summary", {}).get("most_common_content_type"),
                 "common_themes": competitor_data.get("common_themes", [])[:5], # Limit themes
                 "serp_features_present": list(set(sum(serp_data.get("features", {}).values(), []))), # Unique features across all keywords
                 "keywords_with_features_count": len(serp_data.get("features", {})),
                 "top_competitor_domains": list(self._analyze_competitive_landscape_basic(serp_data, competitor_data).get("domain_distribution", {}).keys())[:5],
                 "num_urls_analyzed": len(competitor_data.get("analyzed_urls", [])),
                 # Include counts/summary of generated insights
                 "content_opportunities_count": len(current_insights.get("content_opportunities", [])),
                 "keyword_recommendations_count": len(current_insights.get("keyword_recommendations", [])),
                 "serp_feature_insights_count": len(current_insights.get("serp_feature_insights", [])),
                 "content_blueprints_count": len(current_insights.get("content_blueprints", {})),

             }

             prompt_template = ChatPromptTemplate.from_messages([
                 ("system", """You are an expert SEO and content strategist. Your task is to analyze provided keyword research and competitive analysis data to generate a concise, insightful executive summary.
                 Highlight the main search intent, competitive landscape characteristics (dominant content types, average length, top competitors), key opportunities identified (content gaps, specific recommendations, SERP features), and overall potential for ranking.
                 Focus on the most important findings. Keep the summary brief (around 3-5 sentences).
                 Also, list a few top opportunities (2-3 points) that the user should focus on next.
                 Format the output EXCLUSIVELY as a single, valid JSON object matching the InsightSummary Pydantic model schema.
                 {format_instructions}
                 """),
                 ("human", """Analyze the following data and provide an executive summary and key opportunities:
                 Data: {context}
                 """)]
             )

             chain = prompt_template | self.llm | self.insight_summary_parser

             # Use invoke directly for a single completion
             response = await invoke_with_retry(chain, {"context": json.dumps(context), "format_instructions": self.insight_summary_parser.get_format_instructions()})
             return response.dict() # Return as dictionary

         except Exception as e:
             logger.error(f"Error generating LLM summary: {str(e)}", exc_info=True)
             # Fallback to static summary on error
             return {"overall_summary": self._generate_static_summary(keyword_analysis, serp_data, competitor_data, current_insights), "key_opportunities_highlight": []}


    # *** FIX: Added serp_data to the method signature ***
    async def _generate_content_blueprint(self, target_keyword, serp_results_for_kw, detailed_content_analysis_all, keyword_analysis, competitive_summary, serp_data):
         """Generate a content blueprint for a specific keyword using the LLM."""
         if not self.llm_available or not self.blueprint_parser:
              logger.warning("LLM not available or blueprint parser missing, cannot generate content blueprint.")
              return None # Return None if LLM is required

         try:
             # Filter detailed content analysis to only include relevant URLs for this keyword
             # Use URLs that ranked for this keyword AND were analyzed
             ranked_urls = [res.get("link") for res in serp_results_for_kw if res.get("link")]
             analyzed_urls = detailed_content_analysis_all.keys()
             urls_for_blueprint_context = [url for url in analyzed_urls if url in ranked_urls]

             # If no relevant URLs were analyzed, use data from all analyzed URLs as context (up to 5)
             if not urls_for_blueprint_context:
                  urls_for_blueprint_context = list(analyzed_urls)[:5] # Use top 5 analyzed if no direct match
                  logger.warning(f"No analyzed URLs directly ranking for '{target_keyword}'. Using up to top 5 analyzed URLs for blueprint context.")


             relevant_analysis_data = []
             for url in urls_for_blueprint_context:
                 analysis = detailed_content_analysis_all.get(url)
                 if analysis:
                     # Include only relevant fields to keep context size manageable
                     relevant_analysis_data.append({
                         "url": url,
                         "title": analysis.get("title"),
                         "content_type": analysis.get("content_type"),
                         "word_count": analysis.get("word_count"),
                         "key_themes": analysis.get("key_themes")[:5], # Limit themes
                         "main_points_summary": analysis.get("main_points_summary"),
                         "headings": analysis.get("headings")[:7] # Limit headings
                     })


             # Also include PAA and Related Searches relevant to this keyword from the full serp_data
             # *** FIX: Access serp_data which is now passed as an argument ***
             paa_for_kw_from_serp = serp_data.get("paa_questions", {}).get(target_keyword, [])
             related_for_kw_from_serp = serp_data.get("related_searches", {}).get(target_keyword, [])


             # Include relevant keyword clusters
             relevant_clusters = [cluster for cluster in keyword_analysis.get("clusters", []) if target_keyword in cluster.get("keywords", [])]
             related_cluster_keywords = []
             for cluster in relevant_clusters:
                  related_cluster_keywords.extend(cluster.get("keywords", []))
             related_cluster_keywords = list(set(related_cluster_keywords)) # Unique keywords

             # Get SV and KD for the target keyword for context
             sv_for_kw = serp_data.get("search_volume", {}).get(target_keyword)
             kd_for_kw = serp_data.get("keyword_difficulty", {}).get(target_keyword)
             target_kw_score_data = keyword_analysis.get("keyword_scores", {}).get(target_keyword, {})


             # Prepare context for the LLM
             context = {
                 "target_keyword": target_keyword,
                 "keyword_metrics": {
                      "search_volume": sv_for_kw,
                      "keyword_difficulty": kd_for_kw,
                      "overall_score": target_kw_score_data.get("score")
                 },
                 "overall_intent": keyword_analysis.get("intent_classification", {}).get(target_keyword, "unknown"),
                 "competitive_content_analysis_samples": relevant_analysis_data, # Use sampled data
                 "competitive_landscape_summary": competitive_summary, # Summary from AI analysis
                 "relevant_paa_questions": paa_for_kw_from_serp,
                 "relevant_related_searches": related_for_kw_from_serp,
                 "related_cluster_keywords": related_cluster_keywords, # Keywords from relevant clusters
             }

             prompt_template = ChatPromptTemplate.from_messages([
                 ("system", """You are an expert content strategist and SEO professional. Your task is to analyze the provided data about a target keyword, competitive SERP results, and competitor content to generate a detailed content blueprint. The blueprint should guide the creation of a high-ranking, user-focused piece of content.

                 Analyze the 'competitive_content_analysis_samples' and 'competitive_landscape_summary' to understand common structures, themes, winning angles, and content types.
                 Consider the 'target_keyword', 'keyword_metrics', 'overall_intent', 'relevant_paa_questions', and 'relevant_related_searches' to ensure the blueprint covers user needs and search intent.
                 Use 'related_cluster_keywords' to identify related sub-topics.

                 Generate:
                 - A recommended content type and a realistic word count range.
                 - A structured outline (H1, H2, H3s) that incorporates common successful structures and addresses key themes and user questions.
                 - For each outline section, list specific sub-topics or themes to cover.
                 - Ideas for unique angles to make the content stand out.
                 - List the specific PAA questions from the input that should be addressed in the content.

                 Return the blueprint EXCLUSIVELY as a single, valid JSON object matching the ContentBlueprint Pydantic model schema. Ensure the JSON is perfectly formatted.
                 {format_instructions}
                 """),
                 ("human", """Generate a content blueprint based on the following data:
                 Data: {context}
                 """)]
             )

             chain = prompt_template | self.llm | self.blueprint_parser

             # Use invoke directly for a single completion
             response = await invoke_with_retry(chain, {"context": json.dumps(context), "format_instructions": self.blueprint_parser.get_format_instructions()})
             return response.dict() # Return as dictionary


         except Exception as e:
             logger.error(f"Error generating LLM content blueprint for '{target_keyword}': {str(e)}", exc_info=True)
             return None # Return None if blueprint generation fails


    async def _summarize_competitive_content_strategy(self, analyzed_urls, detailed_content_analysis, competitor_summary_data, serp_api_features):
        """Summarize the competitive content strategy and identify the 'winning formula' using LLM."""
        if not self.llm_available or not self.competitive_summary_parser or not analyzed_urls:
             logger.warning("LLM not available or no URLs analyzed for competitive summary.")
             return {} # Return empty if no LLM or data

        try:
            # Prepare context for the LLM
            context = {
                "analyzed_urls_count": len(analyzed_urls),
                "competitor_content_summary_stats": competitor_summary_data, # Aggregated stats (avg word count, readability, etc.)
                "common_themes_across_urls": competitor_summary_data.get("common_themes_summary", [])[:7], # Top common themes
                "serp_features_observed_keywords": list(serp_api_features.keys()), # Keywords that showed features
                "unique_serp_features": list(set(sum(serp_api_features.values(), []))), # List of unique features
                # Pass snippets of detailed analysis for AI to synthesize patterns (limit to top 5)
                "sample_competitive_analysis": [{
                    "url": url,
                    "title": analysis.get("title"),
                    "content_type": analysis.get("content_type"),
                    "word_count": analysis.get("word_count"),
                    "key_themes": analysis.get("key_themes")[:5], # Limit themes in sample
                    "main_points_summary": analysis.get("main_points_summary"),
                    "headings": analysis.get("headings")[:7] # Limit headings in sample
                } for url, analysis in list(detailed_content_analysis.items())[:5] if analysis], # Sample top 5 analyzed
            }

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert SEO analyst specializing in competitive content strategy. Your task is to analyze the provided data about top-ranking competitor content and summarize the competitive landscape and identify the "winning formula".

                Analyze the 'competitor_content_summary_stats', 'common_themes_across_urls', and 'sample_competitive_analysis' to identify:
                - Dominant content types and average length/readability.
                - Recurring themes, topics, and angles used by top competitors.
                - Common structural patterns (from headings) and media types.
                - How the presence of 'serp_features_observed' might influence content strategy.

                Based on these patterns, articulate the apparent "winning formula" for ranking in this search landscape in a brief narrative summary.

                Return the summary EXCLUSIVELY as a single, valid JSON object matching the CompetitiveStrategySummary Pydantic model schema. Ensure the JSON is perfectly formatted.
                {format_instructions}
                """),
                ("human", """Analyze the following competitive data and summarize the content strategy and winning formula:
                Data: {context}
                """)]
            )

            chain = prompt_template | self.llm | self.competitive_summary_parser

            # Use invoke directly for a single completion
            response = await invoke_with_retry(chain, {"context": json.dumps(context), "format_instructions": self.competitive_summary_parser.get_format_instructions()})
            return response.dict() # Return as dictionary

        except Exception as e:
            logger.error(f"Error generating LLM competitive summary: {str(e)}", exc_info=True)
            # Fallback to basic analysis on error
            return self._analyze_competitive_landscape_basic(None, None) # Pass None as data might be incomplete


    async def _generate_serp_feature_insights_ai(self, serp_api_features, keywords, keyword_scores):
        """Generate actionable SERP feature insights using LLM."""
        if not self.llm_available or not self.serp_feature_parser or not serp_api_features:
             logger.warning("LLM not available or no SERP features detected for insights.")
             return self._generate_serp_feature_insights_basic(None) # Fallback to basic if no LLM or features

        try:
            # Prepare context for the LLM - List features and keywords where they appear
            feature_occurrence = {}
            for keyword, features in serp_api_features.items():
                if isinstance(features, list):
                    for feature in features:
                        if feature not in feature_occurrence:
                            feature_occurrence[feature] = {"count": 0, "keywords": []}
                        feature_occurrence[feature]["count"] += 1
                        feature_occurrence[feature]["keywords"].append(keyword)

            # Filter for features appearing for at least 1 keyword (previously 2+) - adjust threshold as needed
            notable_features = {f: data for f, data in feature_occurrence.items() if data["count"] >= 1}


            # Add keyword scores for context
            features_context = {
                "notable_features": notable_features, # Only include notable features
                "keyword_scores": keyword_scores # Pass full scores
            }

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert SEO specialist focused on SERP features. Analyze the provided data about observed SERP features for a set of keywords.
                For each notable feature, provide a brief description of its significance and specific, actionable strategies to optimize content or the website to target that feature.
                Consider the keywords where the feature appears and their scores (opportunity/difficulty) when suggesting strategies.
                Keep strategies concise and practical.
                Return a JSON array of insights, where each object in the array matches the ActionableSerpFeatureInsight Pydantic model schema. Only include insights for the notable features provided.
                {format_instructions}
                """),
                ("human", """Analyze the following SERP feature data and provide actionable insights:
                Data: {context}
                """)]
            )

            chain = prompt_template | self.llm | self.serp_feature_parser

            # Use invoke directly for a single completion
            response = await invoke_with_retry(chain, {"context": json.dumps(features_context), "format_instructions": self.serp_feature_parser.get_format_instructions()})

            # Validate that the response is indeed a list of dictionaries/objects
            if isinstance(response, list):
                 # Attempt to convert Pydantic objects to dicts if the parser returns them
                 return [item.dict() if hasattr(item, 'dict') else item for item in response]
            else:
                 logger.warning("LLM did not return a list for SERP feature insights.")
                 return self._generate_serp_feature_insights_basic(serp_data) # Fallback


        except Exception as e:
            logger.error(f"Error generating LLM SERP feature insights: {str(e)}", exc_info=True)
            return self._generate_serp_feature_insights_basic(serp_data) # Fallback to basic on error

    async def _generate_intent_matched_recommendations_ai(self, intent_classification, competitor_summary_data, serp_api_features):
        """Generate intent-matched content recommendations using LLM."""
        if not self.llm_available or not self.intent_recommendation_parser:
             logger.warning("LLM not available, cannot generate intent-matched recommendations.")
             return self._generate_content_opportunities_basic(None, None, None) # Fallback

        try:
            # Prepare context for the LLM
            intent_distribution = self._calculate_intent_distribution_from_dict(intent_classification) # Calculate distribution
            context = {
                "intent_distribution": intent_distribution,
                "competitor_content_summary": competitor_summary_data, # Aggregated stats
                "unique_serp_features": list(set(sum(serp_api_features.values(), []))), # List of unique features observed
                # You could add example keywords for each intent if needed
            }

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert content strategist. Analyze the provided data on keyword intent distribution, competitor content patterns, and observed SERP features.
                Identify potential gaps or opportunities based on whether the dominant competitor content types and SERP features fully align with the primary user intents.
                Provide specific, actionable recommendations for content formats or strategies that would better target underserved intents or leverage opportunities in the competitive landscape.
                Focus on clear strategic recommendations.
                Return a JSON array of recommendations, where each object in the array matches the IntentMatchedRecommendation Pydantic model schema.
                {format_instructions}
                """),
                ("human", """Analyze the following data and provide intent-matched content recommendations:
                Data: {context}
                """)]
            )

            chain = prompt_template | self.llm | self.intent_recommendation_parser

            # Use invoke directly for a single completion
            response = await invoke_with_retry(chain, {"context": json.dumps(context), "format_instructions": self.intent_recommendation_parser.get_format_instructions()})

            if isinstance(response, list):
                 return [item.dict() if hasattr(item, 'dict') else item for item in response]
            else:
                 logger.warning("LLM did not return a list for intent-matched recommendations.")
                 return self._generate_content_opportunities_basic(None, None, None) # Fallback

        except Exception as e:
            logger.error(f"Error generating LLM intent recommendations: {str(e)}", exc_info=True)
            return self._generate_content_opportunities_basic(None, None, None) # Fallback


    # --- ENHANCED RULE-BASED / DATA-DRIVEN METHODS ---

    def _generate_keyword_recommendations_enhanced(self, keyword_analysis, serp_data):
        """
        Generate keyword recommendations based on enhanced scores, PAA, and related searches.
        Can be used as a fallback or primary method if LLM is not used for this.
        """
        logger.info("Generating enhanced keyword recommendations.")
        recommendations = []

        # Use keyword_scores which now include SV/KD and refined calculation
        keyword_scores = keyword_analysis.get("keyword_scores", {})

        # 1. Recommend keywords with high overall score (opportunity relative to difficulty)
        high_score_keywords = sorted(keyword_scores.items(), key=lambda item: item[1].get("score", 0), reverse=True)[:10] # Top 10 by overall score

        for kw, score_data in high_score_keywords:
             if score_data.get("score", 0) > 60: # Recommend if score is above a threshold
                  intent = keyword_analysis.get("intent_classification", {}).get(kw, "unknown")
                  recommendations.append({
                       "type": "high_score", # Changed type from high_opportunity
                       "keyword": kw,
                       "reason": f"High overall score ({score_data.get('score', 0)}/100). SV: {score_data.get('search_volume')}, KD: {score_data.get('keyword_difficulty')}.",
                       "score_data": score_data, # Include full score data
                       "intent": intent
                  })
             else:
                  logger.debug(f"Keyword '{kw}' score too low ({score_data.get('score', 0)}) for high_score recommendation.")


        # 2. Recommend Question Keywords (from PAA)
        question_keywords = keyword_analysis.get("question_keywords", [])
        for question in question_keywords[:7]: # Top 7 PAA questions
            # Check if this question keyword already exists as a high_score recommendation to avoid duplication based on keyword string
            if not any(rec.get("keyword", "").lower() == question.lower() for rec in recommendations):
                 # Find score data if available
                 score_data = keyword_scores.get(question, {"score": 50, "search_volume": None, "keyword_difficulty": None})
                 recommendations.append({
                     "type": "question",
                     "keyword": question,
                     "reason": "Identified as a key question (People Also Ask). Good for informational content and featured snippets.",
                     "score_data": score_data,
                     "intent": keyword_analysis.get("intent_classification", {}).get(question, "informational") # Questions are usually informational
                 })


        # 3. Recommend Keywords from Related Searches (if not already recommended)
        related_searches_data = serp_data.get("related_searches", {})
        unique_related_searches = set()
        # Collect all unique related searches across all keywords
        for kw, related_list in related_searches_data.items():
            if isinstance(related_list, list):
                for related_term in related_list:
                    if related_term and isinstance(related_term, str):
                         unique_related_searches.add(related_term.strip())


        for related in list(unique_related_searches)[:7]: # Consider top 7 unique related searches
             # Check if this related term is already recommended
             if not any(rec.get("keyword", "").lower() == related.lower() for rec in recommendations):
                 # Find score data if available
                 score_data = keyword_scores.get(related, {"score": 50, "search_volume": None, "keyword_difficulty": None})
                 recommendations.append({
                     "type": "related_search",
                     "keyword": related,
                     "reason": "Found in related searches, indicating user interest and potential topics.",
                     "score_data": score_data,
                     "intent": keyword_analysis.get("intent_classification", {}).get(related, "unknown") # Classify intent if possible
                 })


        # Remove potential duplicates based on keyword string (case-insensitive)
        keyword_set = set()
        final_recommendations = []
        for rec in recommendations:
             if rec.get("keyword") and rec["keyword"].lower() not in keyword_set:
                  final_recommendations.append(rec)
                  keyword_set.add(rec["keyword"].lower())


        # Sort final recommendations by score, keeping high_score ones at the top generally
        final_recommendations.sort(key=lambda x: x.get('score_data', {}).get('score', 0), reverse=True)

        return final_recommendations


    # --- BASIC FALLBACK METHODS (If LLM not available or fails) ---

    def _generate_static_summary(self, keyword_analysis, serp_data, competitor_data, insights):
        """Generate a static summary of insights (fallback)."""
        try:
            intent_counts = {}
            intent_distribution = self._calculate_intent_distribution(keyword_analysis)
            # Estimate total keywords analyzed if not available
            total_kws_analyzed = len(keyword_analysis.get('intent_classification', {})) or len(keyword_analysis.get('keyword_scores', {}))
            for intent, percentage in intent_distribution.items():
                intent_counts[intent] = int(percentage * total_kws_analyzed) if percentage else 0

            most_common_intent = max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else "unknown"
            most_common_intent_percent = intent_distribution.get(most_common_intent, 0) * 100

            basic_comp_analysis = self._analyze_competitive_landscape_basic(serp_data, competitor_data)
            content_types = [ct["type"] for ct in basic_comp_analysis.get("dominant_content_types", [])]
            content_type_str = ", ".join(content_types[:2]) if content_types else "various"

            avg_word_count = competitor_data.get("summary", {}).get("avg_word_count", 0)
            avg_readability = competitor_data.get("summary", {}).get("avg_readability_score", "N/A")

            keyword_scores = keyword_analysis.get("keyword_scores", {})
            all_scores = list(keyword_scores.values())
            avg_overall_score = np.mean([data.get("score", 50) for data in all_scores]) if all_scores else 50
            avg_opportunity = np.mean([data.get("opportunity", 50) for data in all_scores]) if all_scores else 50
            avg_difficulty = np.mean([data.get("difficulty", 50) for data in all_scores]) if all_scores else 50


            summary = f"""
Based on the analysis of {total_kws_analyzed} keywords (including '{insights.get('seed_keyword', input_data.get('seed_keyword', ''))}').
Primary User Intent: {most_common_intent.capitalize()} ({most_common_intent_percent:.0f}%).

Competitive Content Overview:
- Dominant Formats: {content_type_str}
- Average Length: {avg_word_count} words
- Average Readability (FKGL): {avg_readability}
- Common Themes: {', '.join(competitor_data.get('common_themes', [])[:3]) or 'None identified'}.

Keyword Landscape:
- Average Opportunity Score: {avg_opportunity:.0f}/100
- Average Difficulty Score: {avg_difficulty:.0f}/100
- Keywords with High Potential (Score > 60): {len([k for k,s in keyword_scores.items() if s.get('score', 0) > 60])} found.
- Question-Based Keywords (PAA): {len(keyword_analysis.get('question_keywords', []))} found.

Key Opportunities identified include:
- Content opportunities based on content gaps and intent.
- SERP features present indicating optimization potential.
- Recommended keywords with high scores or relevance.

See the detailed sections below for specific recommendations and analysis.
            """

            return summary.strip() # Return as string for static fallback

        except Exception as e:
            logger.error(f"Error generating static summary: {str(e)}")
            return "Analysis complete. See detailed results below. Error generating summary."


    def _analyze_competitive_landscape_basic(self, serp_data, competitor_data):
        """Analyze the competitive landscape (basic, fallback)."""
        logger.info("Generating basic competitive landscape analysis.")
        analysis = {
            "domain_distribution": {},
            "content_length_avg": competitor_data.get("summary", {}).get("avg_word_count", 0),
            "avg_readability_score": competitor_data.get("summary", {}).get("avg_readability_score", "N/A"), # Use N/A for readability if missing
            "dominant_content_types": [], # List of {type: count}
            "common_themes": competitor_data.get("common_themes", []),
            "domain_authority_level": "unknown", # Cannot determine without external data
            "content_freshness": "unknown", # Cannot determine accurately
            "content_quality": "unknown" # Cannot determine accurately
        }

        if serp_data and "serp_data" in serp_data:
            domain_counts = {}
            # Count domains based on the total number of organic results collected by SerpCollector
            total_organic_results = sum(len(results) for results in serp_data["serp_data"].values() if isinstance(results, list))


            for keyword, results in serp_data["serp_data"].items():
                 if isinstance(results, list):
                     for result in results:
                         if "link" in result: # Use 'link' from SerpAPI structure
                             domain = self._extract_domain(result["link"])
                             domain_counts[domain] = domain_counts.get(domain, 0) + 1


            # Calculate domain distribution (percentage of total organic results)
            for domain, count in domain_counts.items():
                # Include domains that appear more than once, or always include if total results is small
                 if count > 1 or total_organic_results <= 10: # Be less strict for smaller result sets
                    percentage = (count / total_organic_results) * 100 if total_organic_results > 0 else 0
                    analysis["domain_distribution"][domain] = round(percentage)

            # Sort by percentage
            analysis["domain_distribution"] = dict(sorted(analysis["domain_distribution"].items(), key=lambda item: item[1], reverse=True)[:10]) # Top 10

            # Basic domain authority estimation (simplified, check for major sites)
            major_domains = ["wikipedia.org", "amazon.com", "youtube.com", "facebook.com", "linkedin.com", "twitter.com", "reddit.com",
                             "nytimes.com", "washingtonpost.com", "wsj.com", "forbes.com", "cnn.com", "bbc.com"] # Expanded list
            # Calculate percentage of results from major domains within the top domains identified
            major_domain_results_count = sum(
                 count for domain, count in domain_counts.items() # Use raw counts
                 if any(major in domain for major in major_domains)
             )
            major_domain_percentage_in_results = (major_domain_results_count / total_organic_results) * 100 if total_organic_results > 0 else 0


            if major_domain_percentage_in_results > 40:
                 analysis["domain_authority_level"] = "high"
            elif major_domain_percentage_in_results > 15: # Lower threshold for medium
                 analysis["domain_authority_level"] = "medium"
            else:
                 analysis["domain_authority_level"] = "low"


        if competitor_data and "summary" in competitor_data:
             # Get dominant content types from summary counts
             content_types_counts = competitor_data["summary"].get("content_types", {})
             # Convert count dictionary to list of {type: count}
             analysis["dominant_content_types"] = [{"type": t, "count": c} for t, c in content_types_counts.items()]
             analysis["dominant_content_types"].sort(key=lambda item: item["count"], reverse=True)


        return analysis


    def _generate_serp_feature_insights_basic(self, serp_data):
        """Generate basic SERP feature insights (fallback)."""
        logger.info("Generating basic SERP feature insights.")
        insights = []
        if serp_data and "features" in serp_data:
            feature_counts = {}
            for keyword, features in serp_data["features"].items():
                if isinstance(features, list):
                    for feature in features:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1

            total_keywords_with_serp_data = len(serp_data.get("features", {}))
            # Sort features by count
            sorted_features = sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)


            for feature, count in sorted_features:
                if count >= 1: # Show all detected features, even if only once
                    percentage = round((count / total_keywords_with_serp_data) * 100 if total_keywords_with_serp_data > 0 else 0)
                    description = f"Appears for {percentage}% of keywords."
                    strategy = f"Consider optimizing for the {feature.replace('_', ' ')} feature." # Basic strategy

                    insights.append({
                        "feature": feature,
                        "occurrence": count,
                        "percentage": percentage,
                        "description": description,
                        "strategy": strategy
                    })

        return insights

    def _generate_content_opportunities_basic(self, keyword_analysis, serp_data, competitor_data):
        """Generate basic content opportunities (fallback)."""
        logger.info("Generating basic content opportunities.")
        opportunities = []

        # Check for question-based content opportunities (from PAA)
        if keyword_analysis and keyword_analysis.get("question_keywords"):
            questions = keyword_analysis["question_keywords"][:5]
            if questions:
                 opportunities.append({
                    "type": "question_content",
                    "title": "Create FAQ Content",
                    "description": f"Create FAQ content addressing these top questions: {', '.join(questions)}",
                    "keywords": questions
                })

        # Check for content type gaps based on basic competitor analysis summary
        basic_comp_analysis = self._analyze_competitive_landscape_basic(serp_data, competitor_data)
        if basic_comp_analysis.get("dominant_content_types"):
            # Get the type from the first item in the sorted list
            most_common_type = basic_comp_analysis["dominant_content_types"][0].get("type", "unknown")


            # Suggest alternative content types (basic)
            if most_common_type in ["blog post", "informational article", "guide"]: # Group similar types
                opportunities.append({
                    "type": "content_type_gap",
                    "title": "Explore Different Formats",
                    "description": f"Competitors mainly use {most_common_type}. Consider interactive tools, videos, or comparison content.",
                    "content_type": "alternative"
                })
            elif most_common_type in ["product page", "landing page", "category page"]: # Group similar types
                opportunities.append({
                    "type": "content_type_gap",
                    "title": "Create Informational Content",
                    "description": f"Competitors focus on {most_common_type}. Create guides, articles, or reviews for informational/commercial intent.",
                    "content_type": "informational/review"
                })
            elif most_common_type in ["review", "comparison"]: # Group similar types
                 opportunities.append({
                    "type": "content_type_gap",
                    "title": "Create Comprehensive Guides",
                    "description": f"Competitors focus on {most_common_type}. Consider creating in-depth guides for informational intent.",
                    "content_type": "guide"
                })


        # Basic Featured Snippet opportunity
        if serp_data and serp_data.get("features"):
            featured_snippet_keywords = [kw for kw, features in serp_data["features"].items() if "featured_snippet" in features]
            if featured_snippet_keywords:
                 opportunities.append({
                     "type": "serp_feature_opportunity",
                     "title": "Target Featured Snippets",
                     "description": f"Keywords with featured snippets: {', '.join(featured_snippet_keywords[:3])}. Optimize content structure for direct answers.",
                     "keywords": featured_snippet_keywords[:3]
                 })

        # Basic content depth opportunity
        if competitor_data and competitor_data.get("summary", {}).get("avg_word_count", 0) > 0:
             avg_wc = competitor_data["summary"]["avg_word_count"]
             # Suggest longer if average is short, or potentially shorter if very long
             if avg_wc < 1200: # Lower threshold for "short" average
                  opportunities.append({
                    "type": "depth_opportunity",
                    "title": "Create More In-depth Content",
                    "description": f"Competitor content averages {avg_wc} words. Opportunity for longer, comprehensive content (e.g., 1500+ words).",
                    "strategy": "depth"
                })
             elif avg_wc > 3000: # High threshold for "very long" average
                 opportunities.append({
                    "type": "depth_opportunity",
                    "title": "Consider Concise Content",
                    "description": f"Competitor content averages {avg_wc} words. Opportunity for shorter, focused content as an alternative.",
                    "strategy": "conciseness"
                })


        return opportunities


    # --- HELPER METHODS ---

    def _calculate_intent_distribution(self, keyword_analysis):
        """
        Calculate the distribution of intent across keywords.
        Uses intent_classification from KeywordProcessor.
        """
        intent_counts = {}
        intent_classification_dict = keyword_analysis.get("intent_classification", {})
        total_keywords = len(intent_classification_dict)

        for intent in intent_classification_dict.values():
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        distribution = {}
        for intent, count in intent_counts.items():
            distribution[intent] = round(count / total_keywords, 2) if total_keywords > 0 else 0

        return distribution

    def _calculate_intent_distribution_from_dict(self, intent_classification_dict):
        """Calculate intent distribution from a dictionary of keyword: intent."""
        intent_counts = {}
        total_keywords = len(intent_classification_dict)

        for intent in intent_classification_dict.values():
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        distribution = {}
        for intent, count in intent_counts.items():
            distribution[intent] = round(count / total_keywords, 2) if total_keywords > 0 else 0

        return distribution


    def _extract_domain(self, url):
        """
        Extract domain from a URL - Keep existing method.
        """
        try:
            from urllib.parse import urlparse

            if not url:
                return ""

            url = str(url)
            # Handle common browser-use artifact seen in logs "https:" instead of "https://"
            url = url.replace('"https":', 'https://').replace('"http":', 'http://')
            url = url.replace('"', '') # Remove any leftover quotes

            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            if domain.startswith('www.'):
                domain = domain[4:]
            # Handle googleusercontent.com for YouTube links (can indicate YouTube video)
            if "googleusercontent.com" in domain and "youtube.com" in url:
                 return "youtube.com" # Attribute to YouTube for better analysis


            return domain
        except Exception:
            # Return original URL if parsing fails
            return url